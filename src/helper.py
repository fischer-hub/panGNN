import torch
from src.setup import log
from rich.progress import track
from torch_geometric.data import Data



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
    graph_data.class_balance = None

    return graph_data