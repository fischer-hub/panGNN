### some boilerplate code for GNN design which does not use pytorch geometric though
# Define the GNN model
class GeneHomologyGNN(torch.nn.Module):
    def __init__(self, num_genes, embedding_dim, edge_feature_dim, hidden_dim, output_dim=1):
        super(GeneHomologyGNN, self).__init__()

        # Embedding layer to convert gene IDs to vectors
        self.embedding = torch.nn.Embedding(num_genes, embedding_dim)
        
        # Graph convolutional layers
        self.conv1 = GCNConv(3 * embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for classification based on the graph-level output
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, gene_ids, edge_index, edge_attr, batch, neighbor_dict):
        # Convert gene IDs to embeddings
        gene_embeddings = self.embedding(gene_ids)
        
        # Prepare node features by combining with neighbor embeddings
        node_features = self.prepare_node_features_with_neighbors(gene_embeddings, neighbor_dict)
        
        # Pass through the graph convolution layers
        x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pooling the features across the graph
        x = global_mean_pool(x, batch)
        
        # Classification layer
        out = self.classifier(x)
        return out

    def prepare_node_features_with_neighbors(self, gene_embeddings, neighbor_lst):
        """
        Combine each gene's embedding with its neighbors' embeddings.
        
        Parameters:
        - gene_embeddings: Tensor of shape [num_genes, embedding_dim]
        - neighbor_lst: List where indices are gene integer indices and elems are tuples
          of the form (-i-th-upstream_neighbor,... , ith-downstream_neighbor).
          
        Returns:
        - node_features: Tensor with combined neighbor features for each gene.
        """
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