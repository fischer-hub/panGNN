import pickle, os, torch, logging
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

def map_labels_to_edge_index(edge_index, gene_ids_lst, ribap_groups_dict):
    """Map labels for test dataset from RIBAP result table to respective edge
    position in the edge index.

    Args:
        edge_index
    """

    if os.path.isfile('data/labels.pkl'):
        with open('data/labels.pkl', 'rb') as f:
            log.info(f"Found pickled labels, loading file..")
            label_lst = pickle.load(f)
    else:

        num_genes = len(edge_index[0])
        label_lst = [0] * num_genes

        for edge in track(range(num_genes), description = 'Mapping labels to gene pairs in edge index', transient = True):

            source_gene_int_id = edge_index[0][edge]
            destination_gene_int_id = edge_index[1][edge]
            source_gene_str_id = gene_ids_lst[source_gene_int_id]
            destination_gene_str_id = gene_ids_lst[destination_gene_int_id]
            
            if source_gene_str_id in ribap_groups_dict and ribap_groups_dict[source_gene_str_id] == destination_gene_str_id:
                label_lst[edge] = 1
            elif destination_gene_str_id in ribap_groups_dict and ribap_groups_dict[destination_gene_str_id] == source_gene_str_id:
                label_lst[edge] = 1

        if not os.path.isfile('data/labels.pkl'):
            log.info(f"Dumping labels list to pickle file..")
            with open('data/labels.pkl', 'wb') as f:
                pickle.dump(label_lst, f)

    return torch.tensor(label_lst).float()


def load_ribap_groups(ribap_group_file, genome_name_lst):
    """Loads the ribap group file (tab seperated) and returns a pandas dataframe.

    Args:
        ribap_group_file (string, path object): Filename of the file containing the ribap groups.
    """

    ribap_groups_dict = {}
    
    with open(ribap_group_file) as ribap_file_handle:

        ribap_groups_df = pd.read_csv(ribap_file_handle, comment = '#', sep = '\t', header = 0)
        #ribap_groups_df.set_index(ribap_groups_df.columns[0], inplace = True)
        ribap_groups_df.drop(ribap_groups_df.columns.difference(genome_name_lst), axis = 1, inplace=True)
        ribap_groups_df.dropna(inplace=True)


    with Console().status("Constructing two way mapping for ortholog genes..") as status:
        # not sure if this is the best way to get random access but oh well I never claimed to be good at this
        for _, row in ribap_groups_df.iterrows():
            ribap_groups_dict[row[genome_name_lst[0]]] = row[genome_name_lst[1]]
            ribap_groups_dict[row[genome_name_lst[1]]] = row[genome_name_lst[0]]

    return ribap_groups_dict


def combine_neighbour_embeddings(gene_embeddings, neighbor_lst):
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
        upstream_feature= gene_embeddings[upstream_id] if upstream_id is not None else torch.zeros(embedding_dim)
        downstream_feature= gene_embeddings[downstream_id] if downstream_id is not None else torch.zeros(embedding_dim)
        
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

    Keyword arguments:

    edge_index -- The edge index defining the nodes that are connected by each edge.

    bit_score_dict -- The dictionary that holds for every pair of genes the according similarity bit score.

    gene_ids_lst -- The list holding all gene IDs as strings, where the index of an ID in the list is its integer index.
    """

    if os.path.isfile('data/edge_features.pkl'):
        with open('data/edge_features.pkl', 'rb') as f:
            log.info(f"Found pickled edge features, loading file..")
            edge_weight_lst = pickle.load(f)
    else:

        edge_weight_lst = []
        count = 0

        with Console().status("Mapping edge weights to respective edge index positions..") as status:
            for source_int_ID, target_int_ID in zip(edge_index[0], edge_index[1]):
                #print(count / len(edge_index[1]) * 100, ' %')
                #count = count+1
                
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
                        edge_weight_lst.append(0)
                        #print(f"Could not find gene pair in similarity score dataframe, assigning score 0.")

    
    # pickle test data edge features for testing (mapping takes a while otherwise)
    if not os.path.isfile('data/edge_features.pkl'):
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