import pickle, os, torch
import pandas as pd
from src.header import bcolors

def construct_neighour_lst(num_genes: int, num_neighbours: int = 1):
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

    print(f"{bcolors.OKGREEN}Constructed neighbours, first entry: {neighour_lst[0]}{neighour_lst[1]}{neighour_lst[2]}: last entry {neighour_lst[-1]}; length: {len(neighour_lst)-1} {bcolors.ENDC}\n")
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
            print(f"{bcolors.OKGREEN}Found pickled edge features, loading..{bcolors.ENDC}")
            edge_weight_lst = pickle.load(f)
    else:

        edge_weight_lst = []
        count = 0

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
        print(f"{bcolors.OKGREEN}Dumping edge feature list to pickle file..{bcolors.ENDC}")
        with open('data/edge_features.pkl', 'wb') as f:
            pickle.dump(edge_weight_lst, f)
    
    # cast to float since edge weights have to be floats?
    edge_weight_ts = torch.tensor(edge_weight_lst).float()

    print(f"{bcolors.OKGREEN}Successfully created edge feature list with tensor elem type: {edge_weight_ts.dtype}{bcolors.ENDC}")
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