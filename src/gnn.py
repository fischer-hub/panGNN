import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# GCN class based on the example discussed in the pytorch geometric docs
class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim):
        super().__init__()
        
        # input dim is dim of feature vector (embedding)
        # TODO: change dims to enable matrix multiplication in forward step!
        self.conv1 = GCNConv(dataset.node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.node_feature_dim)

    def forward(self, data):

        # NOTE data.edge_attr only contains a tensor with bit scores so basically an edge weight
        # edge attributes can be multiple features embedded, not only a scalar but GCNConv 
        # only takes edge weights (scalars)
        nodes, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        
        nodes = self.conv1(nodes, edge_index, edge_weights)
        nodes = F.relu(nodes)
        #nodes = F.dropout(nodes, training=self.training) # what does this do??
        nodes = self.conv2(nodes, edge_index, edge_weights)

        return nodes #F.log_softmax(nodes, dim=1) # i think this again reduces to one value which we dont want
    
    def decode(self, z, edge_index):
        # calculate dot product between pairs of node embeddings to predict links
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)