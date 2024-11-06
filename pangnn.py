import torch, os
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd

def load_gff(annotation_file_name):
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
    
    return sim_score_df
        



def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


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

    def prepare_node_features_with_neighbors(self, gene_embeddings, neighbor_dict):
        """
        Combine each gene's embedding with its neighbors' embeddings.
        
        Parameters:
        - gene_embeddings: Tensor of shape [num_genes, embedding_dim]
        - neighbor_dict: Dictionary where keys are gene indices and values are tuples
          of the form (upstream_neighbor, downstream_neighbor).
          
        Returns:
        - node_features: Tensor with combined neighbor features for each gene.
        """
        num_genes, embedding_dim = gene_embeddings.shape
        combined_features = []

        for gene_idx in range(num_genes):
            # Get the current gene's embedding
            gene_feat = gene_embeddings[gene_idx]
            
            # Get neighbors' embeddings
            upstream_idx, downstream_idx = neighbor_dict.get(gene_idx, (None, None))
            upstream_feat = gene_embeddings[upstream_idx] if upstream_idx is not None else torch.zeros(embedding_dim)
            downstream_feat = gene_embeddings[downstream_idx] if downstream_idx is not None else torch.zeros(embedding_dim)
            
            # Concatenate embeddings (gene + upstream + downstream)
            combined_feat = torch.cat([upstream_feat, gene_feat, downstream_feat], dim=0)
            combined_features.append(combined_feat)

        return torch.stack(combined_features)

### ENTRY POINT ###
# load annotations from gff files and format to pandas dataframe
genome1_annotation_df = load_gff(os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff'))
print(f"Loaded annotation file of first genome:\n{genome1_annotation_df.head()}")
genome2_annotation_df = load_gff(os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff'))
print(f"Loaded annotation file of second genome:\n{genome1_annotation_df.head()}")

# total number of genes found in all annotation files
num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df)
print(f"Total number of genes found in annotation files: {num_genes}")

# load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
sim_score_df = load_similarity_score(os.path.join('data', 'minimal_mmseq2_result.csv'))
print(f"Loaded similarity scores file:\n{sim_score_df.head()}")

# low embedding dim will reduce risk of overfitting but may prevent model form learning nuanced patterns
# we later concat the embeddings of each gene ID (node) with its neighbouring gene ID embedding -> 3 * embedding dim
gene_id_embedding_dim = 64

# more dims
hidden_dim = 64

# this (the edge features) later holds the similarity bit score of MMSeqs2 clustering for each two gene nodes connected by an edge
edge_feature_dim = 10


#print([list(genome1_annotation_df.index) + list(genome2_annotation_df.index)])

# tensor holding the gene IDs for each gene in the annotation file, note tensors can not be created on strings (or lists of strings)

# map string gene IDs to integer IDs and save in tensor, then embed the int IDs in vector with embedding layer
gene_ids_lst = list(genome1_annotation_df.index) + list(genome2_annotation_df.index)
gene_id_integer_dict = {gene: idx for idx, gene in enumerate(gene_ids_lst)}
gene_ids_ts = torch.tensor(list(gene_id_integer_dict.values()))
print(f"Tensor holding integer encoding of gene IDs: \n{gene_ids_ts}")

# index specifying which nodes are connected by an edge in the graph, e.g.:
# edge_index = torch.tensor([[0, 1],  # source nodes
#                           [1, 2]],  # destination nodes
# dtype=torch.long)  # Edge connections
# however since we have an all-to-all (fully) connected graph where every node (gene)
# has a sequence similarity score to every other node in the graph we define the
# fully connected edge index as follows (tensor magic happening):
row = torch.arange(num_genes).repeat(num_genes)
col = row.view(num_genes, num_genes).t().flatten()
edge_index = torch.stack((row, col), dim=0)
print(f"\nEdge index for fully connected graph successfully created: \n{edge_index}")

# define the edge features (similarity bit scores from MMSeqs2)
# TODO: map the bit score data to the integer IDs (which right now is just the 
# index of the element in the dict luckily), then for 
print(edge_index[0][0])
print(edge_index[1][1])
edge_attr = torch.randn((num_genes/2, edge_feature_dim))  # Edge features
batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

# Neighbor dictionary with each gene's upstream and downstream neighbors
neighbor_dict = {
    0: (None, 1),       # Gene 0 has no upstream neighbor and gene 1 as downstream
    1: (0, 2),          # Gene 1 has gene 0 upstream and gene 2 downstream
    # Add all genes' neighbor info here
}

# Initialize model, optimizer, and loss function
model = GeneHomologyGNN(num_genes=num_genes, embedding_dim=gene_id_embedding_dim, 
                        edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(gene_ids_ts, edge_index, edge_attr, batch, neighbor_dict)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Prediction example
model.eval()
with torch.no_grad():
    pred = model(gene_ids_ts, edge_index, edge_attr, batch, neighbor_dict)
