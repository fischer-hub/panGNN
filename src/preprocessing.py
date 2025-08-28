import pickle, os, torch, time
import pandas as pd
from rich.progress import track, Progress, Console
from src.setup import log, args
import numpy as np
from scipy.special import logsumexp
from collections import Counter
from src.helper import nested_len

def build_adjacency_vectors(num_neighbours, gene_id_lst):

    vector_lst = []

    for gene_id in track(gene_id_lst, transient = True, description = 'Building adjacency vectors..'):
        vector = torch.tensor([0] * len(gene_id_lst), dtype = torch.float32)
        for i in range(-num_neighbours, num_neighbours+1):
            if (gene_id + i) > 0:
                vector[i] = 1
            else:
                continue
        vector_lst.append(vector)
    
    return torch.stack(vector_lst)


def generate_neighbour_edge_features(neighbour_lst, edge_index, sim_score_dict, gene_id_lst):

    neighbour_edge_weights = []
    neighbour_lst = list(neighbour_lst)
    score_default = 1

    # TODO:is the range here correct?
    for origin_gene_id, target_gene_id in zip(edge_index[0], edge_index[1]):

        score = score_default
        
        neighbour_tpl1 = neighbour_lst[origin_gene_id.item()]
        neighbour_tpl2 = neighbour_lst[target_gene_id.item()]
        
        for gene_id1, gene_id2 in zip(neighbour_tpl1, neighbour_tpl2):
            
            # one of the two gene IDs is None type and we can not look it up without crashing
            if not gene_id1 or not gene_id2:
                continue
            # this will never be true if we exclude self edges since the neighbours are from different species and we have unique gene ids fpr each species
            # we have to compare the similarity of the neighbours maybe here. something like the mean sim score of the neighbours maybe?
            #if gene_id1 == gene_id2:
            #    print(origin_gene_id, neighbour_tpl1, target_gene_id, neighbour_tpl2)
            #    score += (1/len(neighbour_tpl1))

            # do another one of the slowest score lookups of all time to get the sim score of the neighbours of our current nodes connected by the edge
            gene_1_str_id = gene_id_lst[gene_id1]
            gene_2_str_id = gene_id_lst[gene_id2]

            try:
                score += sim_score_dict[gene_1_str_id][gene_2_str_id]
            
            except KeyError:
                try:
                    score += sim_score_dict[gene_2_str_id][gene_1_str_id]
                except KeyError:
                    score += 0

        # save the average similarity bit score of the neighbouring genes of the two genes connected by this edge
        if score != 1: 
            score = (score/len(neighbour_tpl1))

        neighbour_edge_weights.append(score)

    return torch.tensor(neighbour_edge_weights)
    

def build_edge_index(sim_score_dict, gene_id_integer_dict, fully_connected = False, self_loops = False):
    """Build an edge index for partially or fully connected input graph from similarity scores and gene IDs.

    Args:
        sim_score_dict (dict): python dictionary containing for every pair of genes in the MMSeqs2 output a similarity score
        gene_id_integer_dict (dict): python dictionary containing for every string gene ID its integer ID counterpart
        fully_connected (bool, optional): bool indicating whether or not to build an edge index for a fully connected graph, that is every node has an edge to every other node in the graph. Defaults to False.
        self_loops (bool, optional): bool indicating wheterh or not an edge from a node to itself is added to the edge index. Defaults to False.

    Returns:
        pytorch tensor: pytorch geometric edge index 
    """

    if fully_connected:
        num_genes = len(list(gene_id_integer_dict.keys()))
        row = torch.arange(num_genes).repeat(num_genes)
        col = row.view(num_genes, num_genes).t().flatten()
        mask = (row != col)
        edge_index_ts = torch.stack((row, col), dim=0) if self_loops else torch.stack((row[mask], col[mask]), dim=0) 
    else:
        max_edge_number = ((len(sim_score_dict)+1) * max(40, (len(sim_score_dict)+1)))
        origin_idx = [None] * max_edge_number
        target_idx = [None] * max_edge_number
        edge_counter = 0

        for origin_id in sim_score_dict:
            for target_id in sim_score_dict[origin_id]:
                # skip self loop edges if self_loops is false
                if not self_loops and (target_id == origin_id):
                    continue
                # when using part of the input genomes from ribap we might have gene pairs in the sim_score_dict 
                # that are not actually loaded in the gene dict, this probably slows down the edge index creation tho
                # so we might need to remove this check once we have a model / use the full dataset as input..
                if target_id in gene_id_integer_dict:
                    origin_idx[edge_counter] = gene_id_integer_dict[origin_id]
                    target_idx[edge_counter] = gene_id_integer_dict[target_id]
                    edge_counter += 1

        #edge_index_ts = torch.stack((torch.tensor(origin_idx), torch.tensor(target_idx)), dim=0)
        origin_idx = origin_idx[:edge_counter]
        target_idx = target_idx[:edge_counter]
        #undirected_origin_idx = origin_idx + target_idx
        #undirected_target_idx = target_idx + origin_idx


    return (origin_idx, target_idx)


# TODO: slow
def map_labels_to_edge_index(edge_index, gene_ids_lst, ribap_groups_dict, use_cache = False):
    """Map labels for test dataset from RIBAP result table to respective edge
    position in the edge index.

    Args:
        edge_index
    """
    if args.cache and use_cache and not os.path.isfile('data/labels.pkl'):
        with open('data/labels.pkl', 'rb') as f:
            log.info(f"Found pickled labels, loading file..")
            label_lst = pickle.load(f)
    else:

        num_genes = len(edge_index[0])
        label_lst = [0] * num_genes

        for edge in range(num_genes):

            source_gene_int_id = edge_index[0][edge]
            destination_gene_int_id = edge_index[1][edge]
            source_gene_str_id = gene_ids_lst[source_gene_int_id]
            destination_gene_str_id = gene_ids_lst[destination_gene_int_id]
            
            if source_gene_str_id in ribap_groups_dict and destination_gene_str_id in ribap_groups_dict[source_gene_str_id]:
                label_lst[edge] = 1
            elif destination_gene_str_id in ribap_groups_dict and source_gene_str_id in ribap_groups_dict[destination_gene_str_id]:
                label_lst[edge] = 1

        if args.cache and use_cache and not os.path.isfile('data/labels.pkl'):
            log.info(f"Dumping labels list to pickle file..")
            with open('data/labels.pkl', 'wb') as f:
                pickle.dump(label_lst, f)
    
    #log.info(f"{sum(label_lst) / len(label_lst)} of edges in ground truth are positive.")
    return torch.tensor(label_lst).float()
    

def load_ribap_groups(ribap_group_file, genome_name_lst):
    """Loads the ribap group file (tab seperated) and returns a pandas dataframe.

    Args:
        ribap_group_file (string, path object): Filename of the file containing the ribap groups.
    """
    ribap_groups_dict = {}
    ribap_groups_lst = []
    
    log.info(f"Loading RIBAP groups file: {ribap_group_file}")

    with open(ribap_group_file) as ribap_file_handle:
        
        ribap_groups_df = pd.read_csv(ribap_file_handle, comment = '#', sep = '\t', header = 0)
        is_subset = not ribap_groups_df.columns.difference(genome_name_lst).empty
        ribap_groups_df.drop(ribap_groups_df.columns.difference(genome_name_lst), axis = 1, inplace=True)

    
    for _, row in track(ribap_groups_df.iterrows(), transient = True, description = "Constructing two way mapping for ortholog genes.."):

        ribap_groups_lst.append([gene for gene in row if isinstance(gene, str)])

        for key_gene in row:
            if not isinstance(key_gene, str):
                continue

            assert key_gene not in ribap_groups_dict, f'{key_gene} already in gene family {ribap_groups_dict[key_gene]}, but also found in gene family {row}.'
            ribap_groups_dict[key_gene] = [gene for gene in row if isinstance(gene, str) and key_gene not in gene]

    assert len(ribap_groups_df.index) <= len(ribap_groups_dict), f'Found {len(ribap_groups_df.index)} gene families in RIBAP file but only {len(ribap_groups_dict)} in RIBAP dictionary.'

    for homologs_lst in ribap_groups_dict.values():
        assert len(homologs_lst) == len(set(homologs_lst)), f'Gene family contains one gene more than once but a gene can be not a homolog to itself: {homologs_lst}'

    return (ribap_groups_dict, ribap_groups_lst, is_subset)


def combine_neighbour_embeddings(gene_embeddings, neighbor_lst, device):
    """
    Combine each gene's embedding with its neighbors' embeddings.
    
    Args:
        gene_embeddings (tensor): Tensor of shape [num_genes, embedding_dim]
                                  take into account.
        neighbor_lst (list):      List where indices are gene integer indices and elems are tuples
                                  of the form (-i-th-upstream_neighbor,... , ith-downstream_neighbor).
        
    Returns:
        node_features (tensor):   Tensor with combined neighbor features for each gene.
    """
    log.debug(f"Got gene embeddings of shape: {gene_embeddings.shape}\n{gene_embeddings}")
    num_genes, embedding_dim = gene_embeddings.shape
    combined_features = []

    for gene_id in range(num_genes):
        # Get the current gene's embedding
        gene_feature = gene_embeddings[gene_id]
        
        # TODO: adjust this to work with more than 1 neighbour
        # get neighbors' embeddings
        upstream_id, downstream_id = neighbor_lst[gene_id]
        upstream_feature= gene_embeddings[upstream_id] if upstream_id is not None else torch.zeros(embedding_dim, device = device)
        downstream_feature= gene_embeddings[downstream_id] if downstream_id is not None else torch.zeros(embedding_dim, device = device)

        # Concatenate embeddings (upstream gene + gene + downstream gene)
        combined_feature= torch.cat([upstream_feature, gene_feature, downstream_feature], dim=0)
        combined_features.append(combined_feature)

    return torch.stack(combined_features)


def construct_neighbour_lst(num_genes: int, num_neighbours: int = 1):
    """Construct list where the index represents a integer gene ID and the tuple
    in that index is contains the predecessor(s) and successor(s) gene of the gene at 
    current index. The first gene hast a predecessor of type None, the last gene
    a succesor of type None.

    Args:
        num_genes        (int): The number of genes to construct neighbours for.
        num_neighbours   (int): The number of predecessor and successor genes to
                                take into account.
    """

    neighour_lst = []

    # TODO:is the range here correct?
    for gene_id in range(num_genes+1):
    
        neighbours = []

        for neighbour in range(-num_neighbours, num_neighbours+1):
            
            # e.g.: [None, None, 1, 2] for num_neighbours = 2, gene_ID = 0
            if neighbour != 0:
                # catch entries without predecessor or successor at ends of genome
                if (gene_id + neighbour) < 0 or (gene_id + neighbour) > num_genes-1:
                    neighbours.append(None)
                else:
                    neighbours.append(gene_id + neighbour)
        
        neighour_lst.append(tuple(neighbours))

    return neighour_lst


def map_edge_weights(edge_index, bit_score_dict, gene_ids_lst, use_cache = False):

    """Returns a tensor that for each node pair in the edge index defines the
    edges weight, that being the similarity bit score of the genes in the two
    nodes connected by the edge. So maps weights to the respective position of 
    its edge in the edge index.

    Args:

        edge_index (tensor): The edge index defining the nodes that are connected by each edge.
        bit_score_dict (dict): The dictionary that holds for every pair of genes the according similarity bit score.
        gene_ids_lst (list): The list holding all gene IDs as strings, where the index of an ID in the list is its integer index.
    
    Returns:
        edge_weight_ts (tensor): tensor that defines for each node pair the 
                                 similarity score of the nodes connected (edges weight)
    """

    if not use_cache and os.path.isfile('data/edge_features.pkl') and args.cache:
        with open('data/edge_features.pkl', 'rb') as f:
            log.info(f"Found pickled edge features, loading file..")
            edge_weight_lst = pickle.load(f)
    else:

        edge_weight_lst = [None] * len(edge_index[0])


        for idx, (source_int_ID, target_int_ID) in enumerate(zip(edge_index[0], edge_index[1])):

            # add pseudo weight 1000 if we have a self loop
            if source_int_ID == target_int_ID:
                edge_weight_lst[idx] = 1000
                continue

            # this should not happen
            if source_int_ID >= len(gene_ids_lst) or target_int_ID >= len(gene_ids_lst):
                edge_weight_lst[idx] = 1
                continue
            else:
                # retrieve str IDs from integer IDs in the edge index
                source_str_ID = gene_ids_lst[source_int_ID]
                target_str_ID = gene_ids_lst[target_int_ID]

            if source_str_ID not in bit_score_dict:
                edge_weight_lst[idx] = 1
            elif target_str_ID not in bit_score_dict[source_str_ID]:
                edge_weight_lst[idx] = 1
            else:
                edge_weight_lst[idx] = bit_score_dict[source_str_ID][target_str_ID]
            
    
    # pickle test data edge features for testing (mapping takes a while otherwise)
    if not use_cache and os.path.isfile('data/edge_features.pkl') and args.cache:
        log.info(f"Dumping edge feature list to pickle file..")
        with open('data/edge_features.pkl', 'wb') as f:
            pickle.dump(edge_weight_lst, f)
    
    
    # cast to float since edge weights have to be floats?
    edge_weight_ts = torch.tensor(edge_weight_lst).float()

    return edge_weight_ts



def load_gff(annotation_file_name, start_gene = 'hemB'):
    """
    Loads an annotation file in GFF format and returns a pandas dataframe.
    """
    with open(annotation_file_name) as gff_handle:

        annotation_df = pd.read_csv(gff_handle, comment = '#', sep = '\t',
                                    names = ['seqname', 'source', 'feature',
                                             'start', 'end', 'score', 'strand',
                                             'frame', 'attribute'],
                                    dtype={'seqname': str, 'source': str, 'feature': str,
                                           'start': 'Int64', 'end': 'Int64', 'score': str,
                                           'strand': str, 'frame': str, 'attribute': str})
        

    start_gene_idx_lst = annotation_df.index[annotation_df['attribute'].str.contains(fr"{start_gene}", na=False)].tolist()

    if start_gene_idx_lst:
        start_gene_idx = start_gene_idx_lst[0]
    else:
        log.error(f"Could not find start gene '{start_gene}' in annotation file, uncentered input genomes might cause falsy gene positions and lead to unstable models.")
        start_gene_idx = 1

    # this works for circular genomes, if we have linear ones we might need to find another solution for an anchor gene
    df1 = annotation_df.iloc[start_gene_idx:, :]
    df2 = annotation_df.iloc[:start_gene_idx, :]

    annotation_df = pd.concat([df1, df2])
    annotation_df.reset_index(drop=True, inplace=True)

    annotation_df = annotation_df.dropna()
    annotation_df['gene_id'] = annotation_df.attribute.str.replace(';.*', '', regex = True)
    annotation_df['gene_id'] = annotation_df.gene_id.str.replace('ID=', '', regex = True)
    
    # filter out non gene IDs, people really put anything into their GFF files huh
    annotation_df = annotation_df[annotation_df['gene_id'].str.contains(r"[A-Z]+_[0-9]+")]
    annotation_df.set_index('gene_id', inplace = True)

    return annotation_df


def remove_trivial_cases(sim_score_dict):

    log.info('Filtering scores for trivial cases..')

    filtered_sim_score_dict = {}

    for source_gene, target_genes in sim_score_dict.items():
        target_gene_ids = target_genes.keys()
        target_genome_ids = [ id.split('_')[0] for id in target_gene_ids ]
        non_unique_target_genome_ids = {k for k, v in Counter(target_genome_ids).items() if v > 1}
        filtered_candidates = { key: val for key, val in target_genes.items() if key.split('_')[0] in non_unique_target_genome_ids }
        if filtered_candidates: filtered_sim_score_dict[source_gene] = filtered_candidates

    log.info(f'Ignoring {len(sim_score_dict) - len(filtered_sim_score_dict)} of {len(sim_score_dict)} scores because they were the only candidate in their candidate set.')
    
    return filtered_sim_score_dict


def load_similarity_score(similarity_score_file, gene_id_position_dict, center_scores = True):

    log.info(f"Loading similarity scores file: {similarity_score_file}")
    with open(similarity_score_file) as sim_score_handle:

        sim_score_df = pd.read_csv(sim_score_handle, comment = '#', sep = '\t', 
                                   names = ['query', 'target', 'pident', 
                                            'alnlen', 'mismatch', 'gapopen', 
                                            'qstart', 'qend', 'qlen', 'tstart', 
                                            'tend', 'tlen', 'qcov', 'tcov', 
                                            'evalue', 'bits'])
    
    sim_score_df = sim_score_df[sim_score_df['query'].isin(gene_id_position_dict)]
    sim_score_df = sim_score_df[sim_score_df['target'].isin(gene_id_position_dict)]

    if center_scores:
        min_score = sim_score_df['bits'].min()
        sim_score_df['bits'] = sim_score_df['bits'] - min_score + 1

    sim_score_df.drop(columns=['pident','alnlen', 'mismatch', 'gapopen',
    #sim_score_df.drop(columns=['bits','alnlen', 'mismatch', 'gapopen',
                               'qstart', 'qend', 'qlen', 'tstart', 
                               'tend', 'tlen', 'qcov', 'tcov', 'evalue'],
                               inplace = True)
    
    sim_score_dict = (
    sim_score_df.groupby('query')
                .apply(lambda x: dict(zip(x['target'], x['bits'])))
                .to_dict())
                # uncomment to replace sim scores with percent identity between genes
                #.apply(lambda x: dict(zip(x['target'], x['pident'])))
    old_len = len(sim_score_dict)
    

    if not args.include_trivial:

        sim_score_dict = remove_trivial_cases(sim_score_dict)

    return sim_score_dict



def softmax_with_temperature(x, t = 0.65):
    """
    Numerically stable softmax using scipy's logsumexp and temperature.
    
    Args:
        x (np.ndarray): Input scores (1D or 2D).
        temperature (float): Temperature parameter.
        
    Returns:
        np.ndarray: Softmax probabilities.
    """
    x = np.asarray(x) / t
    log_denom = logsumexp(x, axis=-1, keepdims=True)
    return np.exp(x - log_denom)


def stable_q_score(p):
    score = -10 * np.log10(np.expm1(np.log1p(-p)))
    if np.isnan(score):
        return 0
    else:
        return score


def normalize_sim_scores(sim_score_dict, t = 0.5, epsilon = 1e-8, pseudo_count = 1, q_score_norm = True):
    
    normalized_dict = {}
    empty_dict_ids = []
    one_count = 0
    all_count = 0

    for origin_gene in track(sim_score_dict.keys(), description = 'Normalizing similarity scores...', transient = True):
        
        candidate_genome_ids = set([id.split('_')[0] for id in sim_score_dict[origin_gene].keys()])
        dict_lst = []
        origin_gene_dict = {}

        for candidate_genome_id in candidate_genome_ids:

            genome_pair_dict = {}

            for candidate_id, score in sim_score_dict[origin_gene].items():
                
                
                if candidate_id.startswith(candidate_genome_id) and candidate_id != origin_gene:
                    # this might cause an underflow in np.exp depending on how
                    # low the score is and how big t is which results in nan in
                    # the division if there are no other scores adding to the sum
                    # in which case the normalized score will be set to 1
                    # this can also overflow when score is to high so we clip it
                    #safe_score = np.clip(t * score, -708, 708)
                    #exp_score = np.exp(safe_score)

                    #if exp_score > (1.7e308 - denominator):
                    #    denominator = 1.7e308
                    #else:
                    #    denominator += exp_score
                    all_count += 1

                    genome_pair_dict.update({candidate_id: score})
            
            score_lst = softmax_with_temperature(list(genome_pair_dict.values()), t) if len(genome_pair_dict) > 1 else [1]
            score_lst = [-10 * np.log10(np.clip(1-prob, epsilon, 1 - epsilon)) if not np.isnan(prob) else -10 * np.log10(1-epsilon) for prob in score_lst]
            #score_lst = [stable_q_score(prob) if not np.isnan(prob) else stable_q_score(epsilon) for prob in score_lst]
            genome_pair_dict = { id: score_lst[i] + pseudo_count for i, id in enumerate(genome_pair_dict) }
            
            # all candidates with the current genome id are in the genome_pair_dict
            # add epsilon for numeric stability
            #for candidate_id, exp_score in genome_pair_dict.items():
            #    if q_score_norm: 
            #        softmax_prob = (exp_score / denominator)
            #    else:
            #        softmax_prob = max((exp_score / denominator), epsilon)
#
            #    if (1-softmax_prob) < 0 and q_score_norm:
            #        print('probability below 0')
            #        softmax_prob = epsilon
            #    # Q score transform
            #    if q_score_norm:
            #        normalized_sim_score = -10 * np.log10(np.clip(1-softmax_prob, epsilon, 1 - epsilon)) if not np.isnan(softmax_prob) else -10 * np.log10(1-epsilon)
            #        #normalized_sim_score = -10 * np.log10(max(1-softmax_prob, epsilon)) if not np.isnan(softmax_prob) else -10 * np.log10(1-epsilon)
            #        assert not np.isnan(normalized_sim_score), f"Found nan after Q score transformation for probability: {1-softmax_prob}, affected Q score: {normalized_sim_score}"
            #        assert np.isfinite(normalized_sim_score), f"Found infinite Q score transformation for probability: {1-softmax_prob}, affected Q score: {normalized_sim_score}"
            #        assert normalized_sim_score != 0, f'Normalized similarity score should never be exactly 0: score: {normalized_sim_score}, num candidates: {genome_pair_dict[candidate_id]}, prob: {softmax_prob}'
            #        genome_pair_dict[candidate_id] = normalized_sim_score + pseudo_count
            #        if genome_pair_dict[candidate_id] == 1.0: one_count +=1
            #        all_count += 1
            #    else:
            #        genome_pair_dict[candidate_id] = softmax_prob if not np.isnan(softmax_prob) else epsilon

            
            # all scores from the current genome are normalized
            dict_lst.append(genome_pair_dict)
        
        # all genomes scores are normalized for this origin gene, merge the dicts
        for d in dict_lst:
            origin_gene_dict.update(d)

        # if dict is empty, skip this gene (e.g. it only holds sim scores to itself)
        if origin_gene_dict:
            normalized_dict[origin_gene] = origin_gene_dict
        else: 
            empty_dict_ids.append(origin_gene)

    # sanity check, is are all genes still in the dict, are all scores in range [0,1]?
    for gene_id in sim_score_dict.keys():
        if gene_id in normalized_dict:
            # length of each candidate dict should be old length -1 since we removed self comparisons
            assert len(sim_score_dict[gene_id]) == len(normalized_dict[gene_id]) + 1 or len(sim_score_dict[gene_id]) == len(normalized_dict[gene_id]), f"Missing normalized score for gene pair ({gene_id}: {normalized_dict[gene_id].keys()}[{len(normalized_dict[gene_id])}], {sim_score_dict[gene_id].keys()}[{len(sim_score_dict[gene_id])}])"
            
            if q_score_norm: 
                assert min(normalized_dict[gene_id].values()) >= pseudo_count, f"Q transformed probability for candidate out of range [pseudo_count, inf) for gene {gene_id}: {normalized_dict[gene_id].values()}"
            else:
                assert min(normalized_dict[gene_id].values()) >= 0 and max(normalized_dict[gene_id].values()) <= 1 + epsilon, f"Probability score for candidate out of range [0,1] for gene {gene_id}: {normalized_dict[gene_id].values()}"


    log.info(f"Normalized similarity scores with t = {t} between gene candidate with loss of {len(empty_dict_ids)} genes in total, e.g. due to only having self comparisons.")
    log.info(f'Created a total of {nested_len(normalized_dict)} edges for input graph.')
    return normalized_dict