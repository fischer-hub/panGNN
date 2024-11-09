import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.preprocessing import combine_neighbour_embeddings, log


# GCN class based on the example discussed in the pytorch geometric docs
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, num_neighbours):
        super().__init__()
        
        # embedding layer for node features
        self.embedding = torch.nn.Embedding(dataset.x.shape[0], dataset.node_feature_dim)

        # input dim is dim of feature vector (embedding) * neighbours (*2 since we have one neighbour in each direction) encoded per node
        # -> embedding vector + (embedding vector * num neighbours * 2)
        combined_embedding_dim = dataset.node_feature_dim + (dataset.node_feature_dim * num_neighbours * 2)
        log.debug(f"Expecting dims {combined_embedding_dim}; {hidden_dim} for first convolution layer.")

        # define convolution layers
        self.conv1 = GCNConv(combined_embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.edge_attr.shape[0])


    def forward(self, data):

        nodes, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        # TODO: does it make sense to call embedding on every forward step? the input doesnt change right?
        # or is this called on the convoluted node embeddings
        node_embeddings = self.embedding(nodes)
        combined_embeddings = combine_neighbour_embeddings(node_embeddings, data.neighbour_lst)

        log.debug(f"Got nodes tensor of shape: {combined_embeddings.shape}")
        log.debug(f"Got edge weights tensor of shape: {edge_weights.shape}")
        log.debug(f"Got edge index of shape: {edge_index.shape}")
        log.debug(f"Node feature embedding (before neighbour combining) dims: {data.node_feature_dim}")
        log.debug(f"Edge feature embedding dims: {data.edge_feature_dim}")

        # NOTE: data.edge_attr only contains a tensor with bit scores so basically an edge weight
        # edge attributes can be multiple features embedded, not only a scalar but GCNConv 
        # only takes edge weights (scalars)
        
        nodes = self.conv1(combined_embeddings, edge_index, edge_weights)
        nodes = F.relu(nodes)
        #nodes = F.dropout(nodes, training=self.training) # what does this do??
        nodes = self.conv2(nodes, edge_index, edge_weights)
        log.debug(f"Outputting nodes of shape: {nodes.shape}\n{nodes}")

        return nodes #F.log_softmax(nodes, dim=1) # i think this again reduces to one value which we dont want
    
    def decode(self, z, edge_index):
        # calculate dot product between pairs of node embeddings to predict links
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)