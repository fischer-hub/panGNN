import torch, random
from src.setup import log, args
from rich.progress import track
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components



def sub_sample_graph_edges(graph, device, fraction = 0.8, sample_pos_edges = False):
    """Subsample graph by sampling from the edge, weight, and label tensors, effectively removing 1-fraction edges from the resulting graph.

    Args:
        fraction (float, optional): Fraction of edges and labels to sample for the resulting graph. Defaults to 0.8.

    Returns:
        PyG Data object: The randomly subsampled graph
    """
    graph.cpu()
    num_neighbour_edges = len(graph.union_edge_index[0]) - len(graph.y)
    
    if sample_pos_edges:
        
        sim_indices = random.sample(range(0, len(graph.y)), int(len(graph.y) * fraction))
        sim_labels = torch.index_select(graph.y, 0, torch.tensor(sim_indices))
    
    # dont leave pos edges behind, sample the counter fraction from edges we leave behind but only sample from the negative indices
    # then use the inverse tensor as batch
    else:

        log.debug('Keeping homolog gene structure intact during graph subsampling.')
        assert (graph.y.sum() / len(graph.y)) <= fraction, f'Trying to subsample {fraction} of the edges in the data when only {graph.y.sum() / len(graph.y)} of edges are positive is not possible when sample_pos_edges is set to False, increase positive edges in data or decrease subsampling fraction'

        negative_labels_indices = list(torch.nonzero(graph.y == 0, as_tuple=True)[0])
        counter_frac_sim_indices = torch.tensor(random.sample(negative_labels_indices, int(len(graph.y) * (1 - fraction))))
        all_indices = torch.arange(len(graph.y))
        inverse_indices = all_indices[~torch.isin(all_indices, counter_frac_sim_indices)]
        sim_indices = list(inverse_indices)
        sim_labels = torch.index_select(graph.y, 0, inverse_indices)
        
        assert (graph.y.sum() - sim_labels.sum()) < 2, f'A total of {graph.y.sum() - sim_labels.sum()} positive edges have been removed during the graph subsampling but at most 2 are allowed when sample_pos_edges is set to False.'

    # maybe we should sample from the additional edges in the union, such that the edges in the sim part are the same in every batch? NO
    union_indices = sim_indices + random.sample(range(len(graph.y), len(graph.y) + num_neighbour_edges), int(num_neighbour_edges * fraction))

    # remap nodes too?
    sim_edge_index_origin = torch.index_select(graph.edge_index[0], 0, torch.tensor(sim_indices)) if not args.union_edge_weights else graph.edge_index[0]
    sim_edge_index_target = torch.index_select(graph.edge_index[1], 0, torch.tensor(sim_indices)) if not args.union_edge_weights else graph.edge_index[1]
    union_edge_index_origin = torch.index_select(graph.union_edge_index[0], 0, torch.tensor(union_indices))
    union_edge_index_target = torch.index_select(graph.union_edge_index[1], 0, torch.tensor(union_indices))
    sim_edge_index = torch.stack((sim_edge_index_origin, sim_edge_index_target))
    union_edge_index = torch.stack((union_edge_index_origin, union_edge_index_target))
    sim_edge_weights = torch.index_select(graph.edge_attr, 0, torch.tensor(sim_indices))
    #union_edge_weights = torch.index_select(graph.union_edge_weight_ts, 0, torch.tensor(union_indices))
    sim_labels = torch.index_select(graph.y, 0, torch.tensor(sim_indices)) if not args.union_edge_weights else graph.y
    #union_labels = torch.index_select(graph.labels_ts, 0, torch.tensor(union_indices))
    
    graph = Data(graph.x, sim_edge_index, sim_edge_weights.float(), sim_labels)
    graph.union_edge_index = union_edge_index
    graph.to(device)
    
    return graph


def concat_graph_data(graph_lst):
    """Concat all graph data objects in the given list into one data object.

    Args:
        graph_lst (list of Data()): list containing the Data() objects to concatenate

    Returns:
        Data(): Data object containing the concatenated graph data objects.
    """

    #x = torch.tensor([])
    idx = 0
    num_graphs = len(graph_lst)
    x, edge_attr, y = [None] * num_graphs, [None] * num_graphs, [None] * num_graphs
    edge_index, union_edge_index = [[None] * num_graphs, [None] * num_graphs], [[None] * num_graphs, [None] * num_graphs]
        
    has_union_index = hasattr(graph_lst[0], 'union_edge_index')
    has_y = True if len(graph_lst[0].y) > 1 else False
    #edge_index = torch.stack((torch.tensor([]), torch.tensor([])))

    #edge_attr = torch.tensor([])
    #union_edge_index = torch.stack((torch.tensor([]), torch.tensor([])))
    #y = torch.tensor([])

    for graph in graph_lst:
        x[idx] = graph.x
        #edge_index = torch.stack((torch.concat((edge_index[0], graph.edge_index[0])), torch.concat((edge_index[1], graph.edge_index[1]))))
        
        edge_index[0][idx] = graph.edge_index[0]
        edge_index[1][idx] = graph.edge_index[1]

        if has_union_index:
            #union_edge_index = torch.stack((torch.concat((union_edge_index[0], graph.union_edge_index[0])), torch.concat((union_edge_index[1], graph.union_edge_index[1]))))
            union_edge_index[0][idx] = graph.union_edge_index[0]
            union_edge_index[1][idx] = graph.union_edge_index[1]
            
        #edge_attr = torch.concat((edge_attr, graph.edge_attr))
        edge_attr[idx] = graph.edge_attr
        
        if has_y is not None:
            #y = torch.concat((y, graph.y))
            y[idx] = graph.y
        else:
            y = None
    
        idx += 1

    x = torch.cat(x)
    edge_attr = torch.cat(edge_attr)
    y = torch.cat(y) if has_y else None


    edge_index = torch.stack((
        torch.cat(edge_index[0]),
        torch.cat(edge_index[1])
    ))


    union_edge_index = torch.stack((
        torch.cat(union_edge_index[0]),
        torch.cat(union_edge_index[1])
    ))
    
    assert has_union_index and len(edge_attr) == len(union_edge_index[0]), f'Number of edges ({len(union_edge_index[0])}) in union edge index does not match number of edge weights ({len(edge_attr)}).'
    assert has_y and len(y) == len(edge_index[0]), f'Number of edges ({len(edge_index[0])}) in similarity edge index does not match number of labels ({len(y)}).'
    
    if has_union_index:
        concated_graph =  Data(x, edge_index.int(), edge_attr, y)
        concated_graph.union_edge_index = union_edge_index.long()
    else:
        concated_graph =  Data(x, edge_index.int(), edge_attr, y)

    return concated_graph



            

def generate_minimal_dataset():

    #nodes = torch.randn(12, 1)
    nodes = torch.tensor([1] * 12).float().unsqueeze(1)

    edge_index_ts = torch.stack((
        torch.tensor([ 0, 1, 2, 3, 4,  6,  7,  8] + [11, 4, 6, 7, 8, 10, 11,  9]),
        torch.tensor([11, 4, 6, 7, 8, 10, 11,  9] + [ 0, 1, 2, 3, 4,  6,  7,  8])
    ))

    union_edge_index_ts = torch.stack((
        torch.cat((edge_index_ts[0], torch.add(torch.tensor([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12] + [2, 3, 1, 3, 4, 1, 2, 4, 2, 3, 6, 7, 5, 7, 8, 5, 6, 8, 6, 7, 10, 11,  9, 11, 12,  9, 10, 12, 10, 11]), -1))),
        torch.cat((edge_index_ts[1], torch.add(torch.tensor([2, 3, 1, 3, 4, 1, 2, 4, 2, 3, 6, 7, 5, 7, 8, 5, 6, 8, 6, 7, 10, 11,  9, 11, 12,  9, 10, 12, 10, 11] + [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12]), -1)))
    ))

    edge_weight_ts = torch.tensor([500, 400, 450, 450, 500, 450, 500, 300] + [500, 400, 450, 450, 500, 450, 500, 300]).float()
    
    labels_ts = torch.tensor([0, 1, 1, 1, 1, 1, 1, 0] + [0, 1, 1, 1, 1, 1, 1, 0]).float()

    graph_data = Data(nodes, edge_index_ts, edge_weight_ts, labels_ts)
    graph_data.union_edge_index = union_edge_index_ts
    graph_data.clss_balance = None

    return graph_data


def simulate_dataset(num_genes, num_genomes, class_balance = 0.2, class_0_stdev = 260, class_0_mean = 212, class_1_stdev = 400, class_1_mean = 550):

    # every genome has num_genes / num_genomes amount of genes
    genome_size = int(num_genes / num_genomes)

    # generate num of nodes as tensor entries
    #nodes = torch.tensor([range(genome_size)] * num_genomes).float().unsqueeze(1)
    id_lst = [list(range(genome_size)) for _ in range(num_genomes)]
    id_lst = [item for sublist in id_lst for item in sublist]
    nodes = torch.tensor(id_lst).float().unsqueeze(1)

    # number of positive edges is defined by the class balance argument
    # this doesnt sclae very good, the amount of edges doesnt scale linearly with the the node number
    # num negative edges = num_possible_edges - num_pos_edges - (num_edges with similarity with score < MMSeqs2 threshold)
    num_edges = num_genes * 30
    num_pos_edges = int(num_edges * class_balance)
    num_negative_edges = num_edges - num_pos_edges

    # for every group of orthologs we need num_genomes-1^2 edges (each gene has a similarity edge to the respective gene in the other genomes)
    # so we get num_pos_edges / num_genomes^2 (just approximate it) number of homolog groups
    num_homolog_groups = int(num_pos_edges / (num_genomes ** 2)) 

    # similarity edges can be between any two nodes (also from the same genome), so draw randomly from the range of genes in input
    # we sample 24 * num_genes edges since this is the ratio I observed in the real klebsiella data
    negative_edge_index = torch.stack((
        torch.tensor(random.choices(range(num_genes), k = num_negative_edges)),
        torch.tensor(random.choices(range(num_genes), k = num_negative_edges))
    ))

    shape = (class_0_mean ** 2) / (class_0_stdev ** 2)
    scale = (class_0_stdev ** 2) / class_0_mean 
    negative_edge_weights = torch.tensor(np.random.gamma(shape = shape, scale = scale, size = num_negative_edges))
    #negative_edge_weights = torch.tensor(random.choices(range(50, 300), k = num_negative_edges))
    negative_labels = torch.tensor([0] * num_negative_edges)

    # create ortholog groups edge index
    origin_nodes = []
    target_nodes = []

    # we can get the gene of same position starting the first genome by multiplying with the genome size (i hope)
    # such that node 1 in the first genome is at the same position as node 1 + genome_size * num_genome in the second


    # TODO: something is off here (maybe all the rounding with int()), the pos edge index seems shorter than the num_pos_edge
    for group in range(num_homolog_groups):
        start_index = random.sample(range(genome_size), 1)[0]

        for start_genome in range(num_genomes):
            for end_genome in range(num_genomes):
                
                # exclude self loops
                if not start_genome == end_genome:
                    origin_nodes.append(start_index + start_genome * genome_size)
                    target_nodes.append(start_index + end_genome * genome_size)


    pos_edge_index = torch.stack((
        torch.tensor(origin_nodes),
        torch.tensor(target_nodes)
    ))

    
    # generate neighbour gene edges
    origin_idx, target_idx = [], []

    # add edges to n nearest neighbour nodes, dont touch the modulo logic it took ages
    for node_idx in range(num_genes):
        for neighbour_id in range(node_idx - args.neighbours, node_idx + args.neighbours + 1):
            if neighbour_id != node_idx:
                # we are k nodes away from genome start, dont add idx - k nodes
                if node_idx % genome_size < args.neighbours:
                    if abs(neighbour_id) % genome_size <= args.neighbours + 1 and neighbour_id >= 0:
                        origin_idx.append(node_idx)
                        target_idx.append(neighbour_id)
                # we are k nodes away from genome end, dont add idx + k nodes
                elif node_idx % genome_size >= (genome_size - args.neighbours):
                    if neighbour_id % genome_size >= args.neighbours:
                        origin_idx.append(node_idx)
                        target_idx.append(neighbour_id)
                # we are in the middle of the genome, add all neighbours
                else:
                    origin_idx.append(node_idx)
                    target_idx.append(neighbour_id)
                


    neighbour_edge_index = torch.stack((
        torch.tensor(origin_idx),
        torch.tensor(target_idx)
    ))

    # quick fix to the difference in expected vs generated pos edges dont tell anyone
    num_pos_edges = len(origin_nodes)

    # paralogs are not of class 1 anyway what did I do here ??
    """
    num_paralogs = int(num_pos_edges * 0.1)
    paralogs_set = 0
    iterations = 0

     while paralogs_set < num_paralogs:

        iterations += 1

        # if we cant find any more high scores, exit
        if iterations > 1000:
            break

        # sample random edge in negative edge set
        i = random.sample(range(len(negative_labels)))

        # check if the sims core is high
        if negative_edge_weights[i] > class_1_mean and negative_labels[i] == 0:
            # if yes, set as paralog
            negative_labels[i] = 1
            paralogs_set += 1 """


    shape = (class_1_mean ** 2) / (class_1_stdev ** 2)
    scale = (class_1_stdev ** 2) / class_1_mean  
    pos_edge_weights = torch.tensor(np.random.gamma(shape = shape, scale = scale, size = num_pos_edges))
    #pos_edge_weights = torch.tensor(random.choices(range(300, 800), k = num_pos_edges))
    pos_labels = torch.tensor([1] * num_pos_edges)

    sim_edge_index = torch.stack((
        torch.cat((negative_edge_index[0], pos_edge_index[0])),
        torch.cat((negative_edge_index[1], pos_edge_index[1]))
    ))
    
    edge_weight_ts = torch.cat((negative_edge_weights, pos_edge_weights)).float()
    labels_ts = torch.cat((negative_labels, pos_labels)).float()


    union_edge_index_ts = torch.stack((
        torch.cat((sim_edge_index[0], neighbour_edge_index[0])),
        torch.cat((sim_edge_index[1], neighbour_edge_index[1])),
    ))


    graph_data = Data(nodes, sim_edge_index, edge_weight_ts, labels_ts)
    graph_data.union_edge_index = union_edge_index_ts
    graph_data.class_balance = class_balance

    log.info(f"{graph_data.y.sum().item() / len(graph_data.y) * 100} % of labels are in positive class.")


    #graph_data.sub_sample_graph_edges = sub_sample_graph_edges.__get__(graph_data, Data)
    #graph_data.__str__ = lambda: 'test'

    return graph_data


def get_connected_nodes(gene_lst, sim_score_dict, n, gene_id_pos_dict, connected_nodes = None):
    """Find n-nearest nodes to nodes from input list that are connected by a similarity edge.

    Args:
        gene_lst (list): list of gene names in input
        sim_score_dict (dict): dictionary holding similarity scores between two genes
        n (int): number of hops to consider from origin node in search
        connected_nodes (set, optional): set of nodes that have already been visited, used for recursion only. Defaults to None.

    Returns:
        list: list of n-nearest nodes connected to the input nodes by a similarity edges
    """

    if connected_nodes is None:
        connected_nodes = set(gene_lst)

    if n == 0:
        return list(connected_nodes)

    new_connected_nodes = set()
    
    for gene in gene_lst:
        
        # check if gene is in list of all genes before checking its in sims core dict, when using a 
        # subset of gff files we can have genes in sim score dict that are not in any gff file
        if gene in gene_id_pos_dict and gene in sim_score_dict:
            new_connected_nodes.update(sim_score_dict[gene].keys())

    new_connected_nodes = new_connected_nodes - connected_nodes
    connected_nodes.update(new_connected_nodes)

    return get_connected_nodes(new_connected_nodes, sim_score_dict, n-1, connected_nodes)



def get_neighbour_graph(gene_lst, gene_id_pos_dict, gene_id_lst, n):

    origin_idx, target_idx = [None] * (len(gene_lst) * n * 2),  [None] * (len(gene_lst) * n * 2)
    edge_counter = 0

    # this will be the remapped node index of neighbour nodes
    neighbour_id_lst = list(gene_lst)
    old_new_pos_dict = {gene_id_pos_dict[gene]: idx for idx, gene in enumerate(gene_lst)}

    for new_origin_gene_pos, origin_gene in enumerate(gene_lst):

        # get position of origin gene in old node index
        old_origin_gene_pos = gene_id_pos_dict[origin_gene]

        # get gene positions of n neighbours in old node index
        for old_neighbour_gene_pos in range(old_origin_gene_pos - n, old_origin_gene_pos + n + 1):

            # check if neighbour position is out of range (start/end of old node index),
            # TODO: we dont check for end of genomes range, but this is at most 2*num_genomes cases
            if old_neighbour_gene_pos < 0 or old_neighbour_gene_pos >= len(gene_id_lst) or old_neighbour_gene_pos == old_origin_gene_pos:
                continue

            # get string id of neighbour gene
            neighbour_gene_id = gene_id_lst[old_neighbour_gene_pos]

            # if we havent seen the string id of the neighbour add it to list (new node index)
            # since we just added the node its edge index id will be len(new node idx) -1
            # map the old position of origin and neighbour nodes to new pos in case we come across the node again
            if neighbour_gene_id not in neighbour_id_lst:
                new_neighbour_gene_pos = len(neighbour_id_lst)-1
                neighbour_id_lst.append(neighbour_gene_id)
                old_new_pos_dict[old_neighbour_gene_pos] = new_neighbour_gene_pos
            else:
                # we have already seen this neighbour use the position it has in the new node index
                new_neighbour_gene_pos = old_new_pos_dict[old_neighbour_gene_pos]

            # add edges between the nodes based on new node index
            origin_idx[edge_counter] = new_origin_gene_pos
            target_idx[edge_counter] = new_neighbour_gene_pos
            edge_counter += 1

    origin_idx = origin_idx[:edge_counter]
    target_idx = target_idx[:edge_counter]
    undirected_origin_idx = origin_idx + target_idx
    undirected_target_idx = target_idx + origin_idx

    edge_index = torch.stack((
        torch.tensor(undirected_origin_idx),
        torch.tensor(undirected_target_idx)
    ))

    gene_id_pos_dict = {gene: idx for idx, gene in enumerate(neighbour_id_lst)}

    # return data object with node features = 1 and labels = 0 (neighbour edges are never label 1)
    return (edge_index, gene_id_pos_dict, neighbour_id_lst)