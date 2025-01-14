import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from src.preprocessing import combine_neighbour_embeddings
from src.setup import log


# GCN class based on the example discussed in the pytorch geometric docs
class MyGCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, num_neighbours, node_feature_dim, device):
        super().__init__()
        self.device = device

        # embedding layer for node features
        #self.embedding = torch.nn.Embedding(len(dataset[0].x)+2, 64)
        self.embedding = torch.nn.Linear(1, 64)

        # input dim is dim of feature vector (embedding) * neighbours (*2 since we have one neighbour in each direction) encoded per node
        # -> embedding vector + (embedding vector * num neighbours * 2)
        combined_embedding_dim = node_feature_dim + (node_feature_dim * num_neighbours * 2)
        log.debug(f"Expecting dims {combined_embedding_dim}; {hidden_dim} for first convolution layer.")

        # define convolution layers
        self.conv_in = GCNConv(64, 128, add_self_loops = True)
        #self.conv2 = DenseGCNConv(128, 128)
        #self.conv2 = GCNConv(128, 128, add_self_loops = True)
        self.conv_hidden = GCNConv(128, 128, add_self_loops = True)
        self.conv_out = GCNConv(128, 64, add_self_loops = True)

        self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, data):

        nodes, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        # TODO: does it make sense to call embedding on every forward step? the input doesnt change right?
        # or is this called on the convoluted node embeddings
        log.debug(f"Got nodes tensor of shape: {nodes.shape}")
        log.debug(f"Got nodes tensor of shape: {nodes}")
        combined_embeddings = self.embedding(nodes)
        log.debug(f"Got nodes tensor after embedding layer of shape: {combined_embeddings.shape}")
        log.debug(f"Got nodes tensor after embedding layer of shape: {combined_embeddings}")
        #combined_embeddings = nodes
        #combined_embeddings = combine_neighbour_embeddings(node_embeddings, data.neighbour_lst, self.device)

        log.debug(f"Got edge weights tensor of shape: {edge_weights.shape}")
        log.debug(f"Got edge index of shape: {edge_index.shape}, {edge_index.dtype}")


        # NOTE: data.edge_attr only contains a tensor with bit scores so basically an edge weight
        # edge attributes can be multiple features embedded, not only a scalar but GCNConv 
        # only takes edge weights (scalars)
        log.debug('Passing data to convolution layer 1..')
        nodes = self.conv_in(combined_embeddings, edge_index, edge_weights)
        log.debug('Passing data to activation function..')
        nodes = F.relu(nodes)
        #nodes = self.activation_fct(nodes)
        #nodes = torch.sigmoid(nodes)
        #nodes = self.conv_hidden(nodes, edge_index, edge_weights)
        #nodes = F.relu(nodes)
        nodes = F.dropout(nodes, training=self.training) # what does this do??
        #nodes = self.conv3(nodes, edge_index, data.neighbour_edge_weights_ts)
        nodes = self.conv_out(nodes, edge_index, edge_weights)
        log.debug(f"Outputting nodes to decode function of shape: {nodes.shape}\n{nodes}")

        link_predictions = self.decode(nodes, edge_index)
        #link_predictions = F.softmax(link_predictions)
        #link_predictions = torch.sigmoid(link_predictions)
        log.debug(f"Outputting link prediction tensor of shape: {link_predictions.shape}\ntype:{link_predictions.dtype}\n{link_predictions}")

        return  link_predictions #F.log_softmax(nodes, dim=1)
    
    def decode(self, z, edge_index):
        # calculate dot product between pairs of node embeddings to predict links
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)