import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from src.preprocessing import combine_neighbour_embeddings
from src.setup import log


# GCN class based on the example discussed in the pytorch geometric docs
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, num_neighbours, node_feature_dim, neighbour_lst):
        super().__init__()

        self.neighbour_lst = neighbour_lst
        
        # embedding layer for node features
        self.embedding = torch.nn.Embedding(dataset.x.shape[0], node_feature_dim)

        # input dim is dim of feature vector (embedding) * neighbours (*2 since we have one neighbour in each direction) encoded per node
        # -> embedding vector + (embedding vector * num neighbours * 2)
        combined_embedding_dim = node_feature_dim + (node_feature_dim * num_neighbours * 2)
        log.debug(f"Expecting dims {combined_embedding_dim}; {hidden_dim} for first convolution layer.")

        # define convolution layers
        self.conv1 = GCNConv(combined_embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, combined_embedding_dim)


    def forward(self, data):

        nodes, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        # TODO: does it make sense to call embedding on every forward step? the input doesnt change right?
        # or is this called on the convoluted node embeddings
        node_embeddings = self.embedding(nodes)
        combined_embeddings = combine_neighbour_embeddings(node_embeddings, self.neighbour_lst)

        log.debug(f"Got nodes tensor of shape: {combined_embeddings.shape}")
        log.debug(f"Got edge weights tensor of shape: {edge_weights.shape}")
        log.debug(f"Got edge index of shape: {edge_index.shape}")

        # NOTE: data.edge_attr only contains a tensor with bit scores so basically an edge weight
        # edge attributes can be multiple features embedded, not only a scalar but GCNConv 
        # only takes edge weights (scalars)
        log.debug('Passing data to convolution layer 1..')
        nodes = self.conv1(combined_embeddings, edge_index, edge_weights)
        log.debug('Passing data to activation layer (ReLU)..')
        nodes = F.relu(nodes)
        log.debug('Passing data to convolution layer 2..')
        #nodes = F.dropout(nodes, training=self.training) # what does this do??
        nodes = self.conv2(nodes, edge_index, edge_weights)
        log.debug(f"Outputting nodes to decode function of shape: {nodes.shape}\n{nodes}")

        link_predictions = self.decode(nodes, edge_index)
        #link_predictions = torch.sigmoid(link_predictions)
        link_predictions = F.relu(link_predictions)
        log.debug(f"Outputting link prediction tensor of shape: {link_predictions.shape}\ntype:{link_predictions.dtype}\n{link_predictions}")

        return  link_predictions #F.log_softmax(nodes, dim=1) # i think this again reduces to one value which we dont want
    
    def decode(self, z, edge_index, binary_th = 0.5):
        # calculate dot product between pairs of node embeddings to predict links
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)