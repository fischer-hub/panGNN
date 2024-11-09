import torch, os, logging, argparse
import torch.nn.functional as F
from torch_geometric.data import Data
from src.header import print_header
from torch_geometric.data import Data
from rich.logging import RichHandler
from rich.progress import track
from rich.traceback import install


###################
###### SETUP ######
###################

print_header(True)

# argparse stuff
parser = argparse.ArgumentParser(
                    prog='pangnn.py',
                    description='The heart and soul of PanGNN. TODO: write sometyhing useful here.',
                    epilog='Greta Garbo and Monroe, Dietrich and DiMaggio, Marlon Brando, Jimmy Dean, On the cover of a magazine.')

#parser.add_argument('annotation_file_name', nargs = '?', default = os.path.join('data', 'Chlamydia_abortus_S26_3_strain_S26_3_full_genome_RENAMED.gff'))           # positional argument
parser.add_argument('-l', '--log_level',  help = "Set the level to print logs ['NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL', Default: 'INFO']", default = 'INFO')
parser.add_argument('-n', '--neighbours', help = 'Number of genes from target gene to consider as neighbours.', default = 1)
parser.add_argument('-d', '--debug',      help = 'Set log level to DEBUG and print debug information while running.', action='store_true')  # on/off flag
args = parser.parse_args()

# setup some terminal formatting and logging
FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level if not args.debug else 'DEBUG', format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
install(show_locals=False)

# we can only import these after setting up the logger since src.preprocessing 
# sets up another logger (for secret reasons) which overwrites the log level..
import src.preprocessing as pp
from src.gnn import GCN


###################
### ENTRY POINT ###
###################


# load annotations from gff files and format to pandas dataframe
genome1_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff'))
log.info(f"Loaded annotation file of first genome: {os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff')}")
genome2_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff'))
log.info(f"Loaded annotation file of second genome: {os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff')}")

# total number of genes found in all annotation files
num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df.index)
log.debug(genome1_annotation_df)
log.info(f"Total number of genes found in annotation files: {num_genes}")

# load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
sim_score_dict = pp.load_similarity_score(os.path.join('data', 'minimal_mmseq2_result.csv'))
log.info(f"Loaded similarity scores file: {os.path.join('data', 'minimal_mmseq2_result.csv')}")

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
log.info(f"Edge index for fully connected graph successfully created.")

# index of the element in the dict luckily), then for 
#torch.set_printoptions(threshold=10_000) # set print limit for tensors
#torch.set_printoptions(profile="full")

# define the edge features (similarity bit scores from MMSeqs2, higher score ~ higher similarity)
edge_weight_ts = pp.map_edge_weights(edge_index_ts, sim_score_dict, gene_ids_lst).reshape([-1, 1])#torch.randn((num_genes/2, edge_feature_dim))  # Edge features
batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

# neighbour dictionary with each gene's upstream and downstream neighbours
neighbour_lst = pp.construct_neighbour_lst(len(genome1_annotation_df.index)) + pp.construct_neighbour_lst(len(genome2_annotation_df.index), num_neighbours = args.neighbours)
log.info(f"Constructed neighbours, first entry: {neighbour_lst[0]}; last entry {neighbour_lst[-1]}; length: {len(neighbour_lst)-1}")


# Initialize model, optimizer, and loss function
#model = GeneHomologyGNN(num_genes=num_genes, embedding_dim=gene_id_embedding_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)

dataset = Data(x = gene_ids_ts, edge_index = edge_index_ts, edge_attr = edge_weight_ts)
dataset.validate()
dataset.node_feature_dim = gene_id_embedding_dim
dataset.edge_feature_dim = edge_feature_dim
dataset.neighbour_lst = neighbour_lst
log.info(f"Constructed dataset from node, egde and index tensors: {dataset}")

model = GCN(dataset = dataset, hidden_dim = hidden_dim, num_neighbours = args.neighbours)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
num_epochs = 10

# Training loop
log.info("Entering training loop..")
for epoch in track(range(num_epochs), description = "Training.."):
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
