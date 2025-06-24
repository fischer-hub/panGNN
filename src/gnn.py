import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from src.preprocessing import combine_neighbour_embeddings
from src.setup import log, args
from torch import nn


# GCN class based on the example discussed in the pytorch geometric docs
class MyGCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, num_neighbours, node_feature_dim, device):
        super().__init__()
        self.device = device

        # embedding layer for node features
        #self.embedding = torch.nn.Embedding(len(dataset[0].x)+2, 64)
        self.embedding = torch.nn.Linear(1, 16)

        # input dim is dim of feature vector (embedding) * neighbours (*2 since we have one neighbour in each direction) encoded per node
        # -> embedding vector + (embedding vector * num neighbours * 2)
        combined_embedding_dim = node_feature_dim + (node_feature_dim * num_neighbours * 2)
        log.debug(f"Expecting dims {combined_embedding_dim}; {hidden_dim} for first convolution layer.")

        # define convolution layers
        self.conv_in = GCNConv(16, 64, add_self_loops = False)
        #self.conv2 = DenseGCNConv(128, 128)
        #self.conv2 = GCNConv(128, 128, add_self_loops = True)
        self.conv_hidden = GCNConv(64, 64, add_self_loops = False)
        self.conv_out = GCNConv(64, 16, add_self_loops = False)

        self.activation_fct = torch.nn.LeakyReLU()


    def forward(self, data):

        nodes, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        # TODO: does it make sense to call embedding on every forward step? the input doesnt change right?
        # or is this called on the convoluted node embeddings
        log.debug(f"Got nodes tensor of shape: {nodes.shape}")
        log.debug(f"Got nodes tensor of shape: {nodes}")
        node_embeddings = self.embedding(nodes)
        log.debug(f"Got nodes tensor after embedding layer of shape: {node_embeddings.shape}")
        log.debug(f"Got nodes tensor after embedding layer of shape: {node_embeddings}")
        #node_embeddings = nodes
        #node_embeddings = combine_neighbour_embeddings(node_embeddings, data.neighbour_lst, self.device)

        log.debug(f"Got edge weights tensor of shape: {edge_weights.shape}")
        log.debug(f"Got edge index of shape: {edge_index.shape}, {edge_index.dtype}")


        # NOTE: data.edge_attr only contains a tensor with bit scores so basically an edge weight
        # edge attributes can be multiple features embedded, not only a scalar but GCNConv 
        # only takes edge weights (scalars)
        log.debug('Passing data to convolution layer 1..')
        nodes = self.conv_in(node_embeddings, edge_index, edge_weights)
        log.debug('Passing data to activation function..')
        nodes = F.relu(nodes)
        #nodes = self.activatineighbourson_fct(nodes)
        #nodes = torch.sigmoid(nodes)
        nodes = self.conv_hidden(nodes, edge_index, edge_weights)
        nodes = F.relu(nodes)
        nodes = self.conv_hidden(nodes, edge_index, edge_weights) 
        nodes = F.relu(nodes)
        #nodes = F.dropout(nodes, training=self.training, p = 0.0001) # what does this do??
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
    



class AlternateGCN(torch.nn.Module):
    def __init__(self, device, dataset, categorical_nodes, dims = [64, 128]):
        super().__init__()
        self.device = device
        node_embedding_dim, hidden_dim = dims

        # embedding layer for node features
        if categorical_nodes:
            # handle gene nodes as categorical data embeddings
            self.embedding = torch.nn.Embedding(len(dataset.x), node_embedding_dim)
            #log.debug(dataset.x)
        else:
            # handle gene nodes as numerical data embeddings
            self.embedding = torch.nn.Linear(1, node_embedding_dim)

        # define convolution layers
        self.conv_in = GCNConv(node_embedding_dim, hidden_dim, add_self_loops = False)
        self.conv_hidden = GCNConv(hidden_dim, hidden_dim, add_self_loops = False)
        self.conv_out = GCNConv(hidden_dim, node_embedding_dim, add_self_loops = False)

        self.linear_out = torch.nn.Linear(hidden_dim, node_embedding_dim)

        #self.activation_fct = torch.nn.LeakyReLU()
        #self.activation_fct = F.relu
        self.activation_fct = torch.nn.ELU()

        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, node_embedding_dim) if not args.skip_connections else nn.Linear((node_embedding_dim * 2) + 1, node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim, 1)
        )

        self.epoch = 0


    def forward(self, graph):
    
        log.debug('Passing nodes to embedding layer..')
        log.debug(graph.x.shape)
        node_embeddings = self.embedding(graph.x)
        log.debug('Passing similarity graph data to convolution layer 1..')

        if args.union_edge_weights:
            nodes = self.conv_in(node_embeddings, graph.union_edge_index, graph.edge_attr)
            nodes = self.activation_fct(nodes)

            for layer in range(max(args.neighbours-2, 1)):
            
                log.debug(f'Passing union graph data to convolution layer {layer + 1}..')
                nodes = self.conv_hidden(nodes, graph.union_edge_index, graph.edge_attr)
                nodes = self.activation_fct(nodes)
            
            nodes = self.conv_out(nodes, graph.union_edge_index)
            nodes = self.activation_fct(nodes)

            #log.debug(f"Outputting nodes to decode function of shape: {nodes.shape}\n{nodes}")

        elif args.base_model:

            graph.sim_edge_index = graph.edge_index

            nodes = self.conv_in(node_embeddings, graph.sim_edge_index, graph.edge_attr)
            nodes = self.activation_fct(nodes)
            nodes = self.linear_out(nodes)
            nodes = self.activation_fct(nodes)


        else:
            
            graph.sim_edge_index = graph.edge_index

            # convolute over similarity edges
            nodes = self.conv_in(node_embeddings, graph.sim_edge_index, graph.edge_attr)
            nodes = self.activation_fct(nodes)

            # convolute over neighbour graph edges
            #nodes = self.conv_hidden(nodes, graph.neighbour_edge_index)
            #nodes = self.activation_fct(nodes)
            
            nodes = self.conv_out(nodes, graph.neighbour_edge_index)
            nodes = self.activation_fct(nodes)

            #log.debug(f"Outputting nodes to decode function of shape: {nodes.shape}\n{nodes}")


        if 'mlp' in args.decoder: 
            if args.skip_connections:
                concat_node_embeddings = torch.cat((nodes[graph.edge_index[0]], nodes[graph.edge_index[1]], graph.edge_attr[:len(graph.edge_index[0])].unsqueeze(1)), dim = 1)
            else:
                concat_node_embeddings = torch.cat((nodes[graph.edge_index[0]], nodes[graph.edge_index[1]]), dim = 1)

            link_predictions = self.mlp(concat_node_embeddings).squeeze(-1)

        if 'cosine' in args.decoder: link_predictions = self.cosine_sim(nodes, graph.edge_index)
        if 'dot' in args.decoder: link_predictions = self.decode(nodes, graph.edge_index)

            #import numpy as np
            #import umap
            #import matplotlib.pyplot as plt
            #umap_reducer = umap.UMAP(n_components=2, random_state=42)
            #embeddings_2d = umap_reducer.fit_transform(nodes)
#
            ## Plot UMAP
            #plt.figure(figsize=(10, 8))
            #plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, cmap='Spectral')
            #plt.title("UMAP of Node Embeddings", fontsize=14)
            #plt.xlabel("UMAP Dimension 1")
            #plt.ylabel("UMAP Dimension 2")
            #plt.colorbar(label="Node IDs")
            #plt.savefig(f'umap_frames/{self.epoch}.png')
            #self.epoch += 1
            #plt.close()        
            #log.debug(f"Outputting link prediction tensor of shape: {link_predictions.shape}\ntype:{link_predictions.dtype}\n{link_predictions}")

        return  link_predictions
    
    def decode(self, z, edge_index):
        # calculate dot product between pairs of node embeddings to predict links
        return z[edge_index[0]] @ z[edge_index[1]]
    
    def cosine_sim(self, z, edge_index):
        return F.cosine_similarity(z[edge_index[0]], z[edge_index[1]], dim = 1)