import torch, os, pickle
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from src.header import print_header, bcolors
import src.preprocessing as pp
from src.gnn import GCN
from torch_geometric.data import Data

        




###################
### ENTRY POINT ###
###################

print_header(True)

# load annotations from gff files and format to pandas dataframe
genome1_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff'))
print(f"{bcolors.OKGREEN}Loaded annotation file of first genome: \n {bcolors.ENDC}{genome1_annotation_df.head()}")
genome2_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff'))
print(f"{bcolors.OKGREEN}Loaded annotation file of second genome: \n {bcolors.ENDC}{genome1_annotation_df.head()}")

# total number of genes found in all annotation files
num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df.index)
print(f"{bcolors.OKGREEN}Total number of genes found in annotation files: {num_genes}")

# load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
sim_score_dict = pp.load_similarity_score(os.path.join('data', 'minimal_mmseq2_result.csv'))
print(f"{bcolors.OKGREEN}Loaded similarity scores file: \n {bcolors.ENDC}")

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

# reshape tensor to have correct dimensions
gene_ids_ts = torch.tensor(list(gene_id_integer_dict.values())).reshape([-1, 1])
print(f"\n{bcolors.OKGREEN}Tensor holding integer encoding of gene IDs:  \n {bcolors.ENDC}{gene_ids_ts}")

# index specifying which nodes are connected by an edge in the graph, e.g.:
# edge_index = torch.tensor([[0, 1],  # source nodes
#                           [1, 2]],  # destination nodes
# dtype=torch.long)  # Edge connections
# however since we have an all-to-all (fully) connected graph where every node (gene)
# has a sequence similarity score to every other node in the graph we define the
# fully connected edge index as follows (tensor magic happening):
row = torch.arange(num_genes).repeat(num_genes)
col = row.view(num_genes, num_genes).t().flatten()
edge_index_ts = torch.stack((row, col), dim=0)
print(f"{bcolors.OKGREEN} \nEdge index for fully connected graph successfully created:  \n {bcolors.ENDC}{edge_index_ts}")

# index of the element in the dict luckily), then for 
#torch.set_printoptions(threshold=10_000) # set print limit for tensors
#torch.set_printoptions(profile="full")

# define the edge features (similarity bit scores from MMSeqs2, higher score ~ higher similarity)
edge_weight_ts = pp.map_edge_weights(edge_index_ts, sim_score_dict, gene_ids_lst)#torch.randn((num_genes/2, edge_feature_dim))  # Edge features
batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

# neighbour dictionary with each gene's upstream and downstream neighbours
neighbour_lst = pp.construct_neighour_lst(len(genome1_annotation_df.index)) + pp.construct_neighour_lst(len(genome2_annotation_df.index))

# Initialize model, optimizer, and loss function
#model = GeneHomologyGNN(num_genes=num_genes, embedding_dim=gene_id_embedding_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)

dataset = Data(x = gene_ids_ts, edge_index = edge_index_ts, edge_attr = edge_weight_ts)
dataset.validate()
dataset.node_feature_dim = gene_id_embedding_dim
dataset.edge_feature_dim = edge_feature_dim
print(f"Constructed dataset from tensors: \n{dataset}")

model = GCN(dataset = dataset, hidden_dim = hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    # clear old gradients so only current batch gradients are used
    optimizer.zero_grad()
    #output = model(gene_ids_ts, edge_index, edge_attr_ts, batch, neighbour_lst)

    # this calls the models forward function since model is callable
    output = model(dataset)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Prediction example
model.eval()
with torch.no_grad():
    pred = model(gene_ids_ts, edge_index, edge_attr, batch, neighbour_lst)
