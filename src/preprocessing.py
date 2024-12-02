import pickle, os, torch, time
import pandas as pd
from rich.progress import track, Progress
from src.setup import log, args


def generate_neighbour_edge_features(neighbour_lst, edge_index):

    neighbour_edge_weights = []
    neighbour_lst = list(neighbour_lst)
    score_default = 1

    # TODO:is the range here correct?
    for origin_gene_id, target_gene_id in zip(edge_index[0], edge_index[1]):

        score = score_default
        
        neighbour_tpl1 = neighbour_lst[origin_gene_id.item()]
        neighbour_tpl2 = neighbour_lst[target_gene_id.item()]
        
        for gene_id1, gene_id2 in zip(neighbour_tpl1, neighbour_tpl2):
            
            if gene_id1 == gene_id2:
                score += (1/len(neighbour_tpl1))
            
        neighbour_edge_weights.append(score)

    if sum(neighbour_edge_weights) == (score_default * len(neighbour_edge_weights)):
        log.warning(f"It seems no two genes in the data have a similar neighbourhood, how odd! I would check the data if I was you ;)")
        time.sleep(5)

    return torch.tensor(neighbour_edge_weights)


def map_edge_weights(edge_index, bit_score_dict, gene_ids_lst):

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

    if os.path.isfile('data/edge_features.pkl') and args.cache:
        with open('data/edge_features.pkl', 'rb') as f:
            log.info(f"Found pickled edge features, loading file..")
            edge_weight_lst = pickle.load(f)
    else:

        edge_weight_lst = []


        with Progress(transient = True) as progress:
            edge_weight_bar = progress.add_task("Mapping edge weights to respective edge index positions..\n", total=len(edge_index[1]))
            #with Console().status("Mapping edge weights to respective edge index positions..") as status:
            for source_int_ID, target_int_ID in zip(edge_index[0], edge_index[1]):

                # add pseudo weight 1000 if we have a self loop
                if source_int_ID == target_int_ID:
                    edge_weight_lst.append(1000)
                    continue
                # retrieve str IDs from integer IDs in the edge index
                source_str_ID = gene_ids_lst[source_int_ID]
                target_str_ID = gene_ids_lst[target_int_ID]

                # look up bit score for string IDs of the two genes and save to list
                #print(f"Starting lookup for source node: ({source_str_ID}, {source_int_ID}); Target node: ({target_str_ID}, {target_int_ID})")

                try:
                    edge_weight = bit_score_dict[source_str_ID][target_str_ID]
                    edge_weight_lst.append(edge_weight)
                    #print(f"Bit score: {edge_weight}")
                
                except KeyError:
                    try:
                        edge_weight = bit_score_dict[target_str_ID][source_str_ID]
                        edge_weight_lst.append(edge_weight)
                        #print(f"Bit score: {edge_weight}")
                    except KeyError:
                        # do we want 0 as no similarity score? does this 'kill' neurons and preevent from learning?
                        edge_weight_lst.append(0)
                        #print(f"Could not find gene pair in similarity score dataframe, assigning score 0.")
                
                progress.update(edge_weight_bar, advance = 1)
            
    
    # pickle test data edge features for testing (mapping takes a while otherwise)
    if not os.path.isfile('data/edge_features.pkl') and args.cache:
        log.info(f"Dumping edge feature list to pickle file..")
        with open('data/edge_features.pkl', 'wb') as f:
            pickle.dump(edge_weight_lst, f)
    
    
    # cast to float since edge weights have to be floats?
    edge_weight_ts = torch.tensor(edge_weight_lst).float()

    log.info(f"Successfully created edge feature list with tensor elem type: {edge_weight_ts.dtype}")
    return edge_weight_ts
    

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
        origin_idx, target_idx = [], []

        for origin_id in sim_score_dict.keys():
            for target_id in sim_score_dict[origin_id].keys():
                # skip self loop edges if self_loops is false
                if not self_loops and (target_id == origin_id):
                    continue
                # when using part of the input genomes from ribap we might have gene pairs in the sim_score_dict 
                # that are not actually loaded in the gene dict, this probably slows down the edge index creation tho
                # so we might need to remove this check once we have a model / use the full dataset as input..
                if origin_id in gene_id_integer_dict.keys() and target_id in gene_id_integer_dict.keys():
                    origin_idx.append(gene_id_integer_dict[origin_id])
                    target_idx.append(gene_id_integer_dict[target_id])

        edge_index_ts = torch.stack((torch.tensor(origin_idx), torch.tensor(target_idx)), dim=0)
    return edge_index_ts



def map_labels_to_edge_index(edge_index, gene_ids_lst, ribap_groups_dict):
    """Map labels for test dataset from RIBAP result table to respective edge
    position in the edge index.

    Args:
        edge_index
    """

    if os.path.isfile('data/labels.pkl') and args.cache:
        with open('data/labels.pkl', 'rb') as f:
            log.info(f"Found pickled labels, loading file..")
            label_lst = pickle.load(f)
    else:

        num_genes = len(edge_index[0])
        label_lst = [0] * num_genes

        for edge in track(range(num_genes), description = 'Mapping labels to gene pairs in edge index..\n', transient = True):

            source_gene_int_id = edge_index[0][edge]
            destination_gene_int_id = edge_index[1][edge]
            source_gene_str_id = gene_ids_lst[source_gene_int_id]
            destination_gene_str_id = gene_ids_lst[destination_gene_int_id]
            
            if source_gene_str_id in ribap_groups_dict and ribap_groups_dict[source_gene_str_id] == destination_gene_str_id:
                label_lst[edge] = 1
            elif destination_gene_str_id in ribap_groups_dict and ribap_groups_dict[destination_gene_str_id] == source_gene_str_id:
                label_lst[edge] = 1

        if not os.path.isfile('data/labels.pkl') and args.cache:
            log.info(f"Dumping labels list to pickle file..")
            with open('data/labels.pkl', 'wb') as f:
                pickle.dump(label_lst, f)
    
    log.info(f"{sum(label_lst) / len(label_lst)} of edges in ground truth are positive.")
    return torch.tensor(label_lst).float()


def load_ribap_groups(ribap_group_file, genome_name_lst):
    """Loads the ribap group file (tab seperated) and returns a pandas dataframe.

    Args:
        ribap_group_file (string, path object): Filename of the file containing the ribap groups.
    """

    ribap_groups_dict = {}
    ribap_groups_dict_tmp = {}
    
    with open(ribap_group_file) as ribap_file_handle:

        ribap_groups_df = pd.read_csv(ribap_file_handle, comment = '#', sep = '\t', header = 0)
        ribap_groups_df.drop(ribap_groups_df.columns.difference(genome_name_lst), axis = 1, inplace=True)


    for i in range(len(genome_name_lst)-1):
        for j in range(1, len(genome_name_lst)):
            # not sure if this is the best way to get random access but oh well I never claimed to be good at this
            for _, row in ribap_groups_df.iterrows():#, description = 'Constructing two way mapping for ortholog genes..', transient = True):
                if not pd.isna(row[genome_name_lst[j]]) and not pd.isna(row[genome_name_lst[i]]):
                    ribap_groups_dict_tmp[row[genome_name_lst[i]]] = row[genome_name_lst[j]]
                    ribap_groups_dict_tmp[row[genome_name_lst[j]]] = row[genome_name_lst[i]]
    
        ribap_groups_dict.update(ribap_groups_dict_tmp)

    return ribap_groups_dict


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


def map_edge_weights(edge_index, bit_score_dict, gene_ids_lst):

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

    if os.path.isfile('data/edge_features.pkl') and args.cache:
        with open('data/edge_features.pkl', 'rb') as f:
            log.info(f"Found pickled edge features, loading file..")
            edge_weight_lst = pickle.load(f)
    else:

        edge_weight_lst = []


        with Progress(transient = True) as progress:
            edge_weight_bar = progress.add_task("Mapping edge weights to respective edge index positions..\n", total=len(edge_index[1]))
            #with Console().status("Mapping edge weights to respective edge index positions..") as status:
            for source_int_ID, target_int_ID in zip(edge_index[0], edge_index[1]):

                # add pseudo weight 1000 if we have a self loop
                if source_int_ID == target_int_ID:
                    edge_weight_lst.append(1000)
                    continue
                # retrieve str IDs from integer IDs in the edge index
                source_str_ID = gene_ids_lst[source_int_ID]
                target_str_ID = gene_ids_lst[target_int_ID]

                # look up bit score for string IDs of the two genes and save to list
                #print(f"Starting lookup for source node: ({source_str_ID}, {source_int_ID}); Target node: ({target_str_ID}, {target_int_ID})")

                try:
                    edge_weight = bit_score_dict[source_str_ID][target_str_ID]
                    edge_weight_lst.append(edge_weight)
                    #print(f"Bit score: {edge_weight}")
                
                except KeyError:
                    try:
                        edge_weight = bit_score_dict[target_str_ID][source_str_ID]
                        edge_weight_lst.append(edge_weight)
                        #print(f"Bit score: {edge_weight}")
                    except KeyError:
                        # do we want 0 as no similarity score? does this 'kill' neurons and preevent from learning?
                        edge_weight_lst.append(0)
                        #print(f"Could not find gene pair in similarity score dataframe, assigning score 0.")
                
                progress.update(edge_weight_bar, advance = 1)
            
    
    # pickle test data edge features for testing (mapping takes a while otherwise)
    if not os.path.isfile('data/edge_features.pkl') and args.cache:
        log.info(f"Dumping edge feature list to pickle file..")
        with open('data/edge_features.pkl', 'wb') as f:
            pickle.dump(edge_weight_lst, f)
    
    
    # cast to float since edge weights have to be floats?
    edge_weight_ts = torch.tensor(edge_weight_lst).float()

    log.info(f"Successfully created edge feature list with tensor elem type: {edge_weight_ts.dtype}")
    return edge_weight_ts



def load_gff(annotation_file_name):
    """
    Loads an annotation file in GFF format and returns a pandas dataframe.
    """
    with open(annotation_file_name) as gff_handle:

        annotation_df = pd.read_csv(gff_handle, comment = '#', sep = '\t', 
                                    names = ['seqname', 'source', 'feature', 
                                             'start', 'end', 'score', 'strand', 
                                             'frame', 'attribute'])

    annotation_df = annotation_df.dropna()
    annotation_df['gene_id'] = annotation_df.attribute.str.replace(';.*', '', regex = True)
    annotation_df['gene_id'] = annotation_df.gene_id.str.replace('ID=', '', regex = True)
    annotation_df.set_index('gene_id', inplace = True)

    return annotation_df


def load_similarity_score(similarity_score_file):
    with open(similarity_score_file) as sim_score_handle:

        sim_score_df = pd.read_csv(sim_score_handle, comment = '#', sep = '\t', 
                                   names = ['query', 'target', 'pident', 
                                            'alnlen', 'mismatch', 'gapopen', 
                                            'qstart', 'qend', 'qlen', 'tstart', 
                                            'tend', 'tlen', 'qcov', 'tcov', 
                                            'evalue', 'bits'])
        
    sim_score_df.drop(columns=['pident','alnlen', 'mismatch', 'gapopen', 
                               'qstart', 'qend', 'qlen', 'tstart', 
                               'tend', 'tlen', 'qcov', 'tcov', 'evalue'],
                               inplace = True)
    
    sim_score_dict = (
    sim_score_df.groupby('query')
                .apply(lambda x: dict(zip(x['target'], x['bits'])))
                .to_dict())
    
    return sim_score_dict