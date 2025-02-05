import torch, random
from src.setup import log, args
from rich.progress import track
import numpy as np
from torch_geometric.data import Data



def sub_sample_graph_edges(graph, device, fraction = 0.8):
    """Subsample graph by sampling from the edge, weight, and label tensors, effectively removing 1-fraction edges from the resulting graph.

    Args:
        fraction (float, optional): Fraction of edges and labels to sample for the resulting graph. Defaults to 0.8.

    Returns:
        PyG Data object: The randomly subsampled graph
    """
    graph.cpu()
    num_neighbour_edges = len(graph.union_edge_index[0]) - len(graph.y)
    sim_indices = random.sample(range(0, len(graph.y)), int(len(graph.y) * fraction))
    # maybe we should sample from the additional edges in the union, such that the edges in the sim part are the same in every batch?
    union_indices = sim_indices + random.sample(range(len(graph.y), len(graph.y) + num_neighbour_edges), int(num_neighbour_edges * fraction))

    # remap nodes too?
    sim_edge_index_origin = torch.index_select(graph.edge_index[0], 0, torch.tensor(sim_indices))
    sim_edge_index_target = torch.index_select(graph.edge_index[1], 0, torch.tensor(sim_indices))
    union_edge_index_origin = torch.index_select(graph.union_edge_index[0], 0, torch.tensor(union_indices))
    union_edge_index_target = torch.index_select(graph.union_edge_index[1], 0, torch.tensor(union_indices))
    sim_edge_index = torch.stack((sim_edge_index_origin, sim_edge_index_target))
    union_edge_index = torch.stack((union_edge_index_origin, union_edge_index_target))
    sim_edge_weights = torch.index_select(graph.edge_attr, 0, torch.tensor(sim_indices))
    #union_edge_weights = torch.index_select(graph.union_edge_weight_ts, 0, torch.tensor(union_indices))
    sim_labels = torch.index_select(graph.y, 0, torch.tensor(sim_indices))
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

    x = torch.tensor([])
    edge_index = torch.stack((torch.tensor([]), torch.tensor([])))
    edge_attr = torch.tensor([])
    y = torch.tensor([])

    for graph in graph_lst:
        x = torch.concat((x, graph.x))
        edge_index = torch.stack((torch.concat((edge_index[0], graph.edge_index[0])), torch.concat((edge_index[1], graph.edge_index[1]))))
        edge_attr = torch.concat((edge_attr, graph.edge_attr))
        
        if graph.y is not None:
            y = torch.concat((y, graph.y))
        else:
            y = None

    edge_index = edge_index.long()
    
    return Data(x, edge_index, edge_attr, y)



# we use a scipy function for this as I am appearantly too dumb to respect all edge cases and this doesnt work..
def separate_components(edge_index):
    """Separate connected components in a given graph based on its edge index.

    Args:
        edge_index (pytorch tensor): tensor containing the edges of the graph to separate
    
    Returns:
        connected components (list of tuples): list containing for each connected component (subgraph)
        a tuple of form ([originial_indices], [origin_node_id], [target_node_id]), where all nodes in the
        tuple are connected by an edge defined in tuple[1], tuple[2] and tuple[0] containing the index
        of that edge in the original edge index (might be equal to the index of e.g. the corresponding edge weight)
    """

    connected_components = []
    
    for idx, (origin_id, target_id) in track(enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())), description = 'Separating connected components in graph..', transient = True):

        found = False
        found_cnt = 0
        # each component is a tuple of lists of form ([], [], []), representing 
        # the edge index of one connected component in the input in pos 1, 2 and
        # the indices i in the old edge index at pos 0
        for comp_idx, component in enumerate(connected_components):

            # we already found a component that this edge belongs to, if we find 
            # another one the two are connected by this edge and should be merged 
            # to a new one
            if found and component[3]:
                old_component = connected_components[comp_idx]
                log.info(f"Edge [{origin_id}, {target_id}] found in component {component} but also occures in previous component {old_component}! Merging components.")

                # tuples are my personal nightmare
                component = (old_component[0] + component[0], old_component[1] + component[1], old_component[2] + component[2])

                connected_components[comp_idx] = (old_component[0], old_component[1], old_component[2], False)
                break

            # check if nodes are part of current component
            elif origin_id in component[1] or origin_id in component[0] or target_id in component[0] or target_id in component[1]:
                found = comp_idx
                component[0].append(idx)
                component[1].append(origin_id)
                component[2].append(target_id)
        
        # we checked all components and none contained this edge, create new component,
        # the last tuple entry is a bool to keep track of components that have been merged 
        # and should be removed afterwards
        if not found: connected_components.append(([idx], [origin_id], [target_id], True))
    
    log.info(f"Found {len(connected_components)} connected components in input data.")
    connected_components = [connected_components for component in connected_components if component[3]]
    log.info(f"Found {len(connected_components)} connected components after merging.")


    return connected_components
            

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

    # generate num of nodes as tensor entries
    nodes = torch.tensor([1] * num_genes).float().unsqueeze(1)

    # every genome has num_genes / num_genomes amount of genes
    genome_size = int(num_genes / num_genomes)

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
            paralogs_set += 1


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

