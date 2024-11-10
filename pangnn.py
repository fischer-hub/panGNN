import torch, os, logging, argparse
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from src.header import print_header
from src.plot import plot_loss_accuracy
from torch_geometric.data import Data
from rich.logging import RichHandler
from rich.progress import track, Progress
from rich.traceback import install
from rich.console import Console



###################
###### SETUP ######
###################

print_header(True)

# argparse stuff
parser = argparse.ArgumentParser(
                    prog='pangnn.py',
                    description='The heart and soul of PanGNN. TODO: write sometyhing useful here.',
                    epilog='Greta Garbo and Monroe, Dietrich and DiMaggio, Marlon Brando, Jimmy Dean, On the cover of a magazine.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('annotation_file_name', nargs = '?', default = os.path.join('data', 'Chlamydia_abortus_S26_3_strain_S26_3_full_genome_RENAMED.gff'))           # positional argument
parser.add_argument('-l', '--log_level',  help = "set the level to print logs ['NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']", default = 'INFO', type = str)
parser.add_argument('-n', '--neighbours', help = 'number of genes from target gene to consider as neighbours', default = 1, type = int)
parser.add_argument('-m', '--model',      help = 'path to save or load model from, depending on train or inference mode', default = 'model.pkl', type = str)
parser.add_argument('-b', '--batch_size', help = 'set dataset batch size for model training', default = 32, type = int)
parser.add_argument('-e', '--epochs',     help = 'set number of epochs for model training', default = 1, type = int)
parser.add_argument('-d', '--debug',      help = 'set log level to DEBUG and print debug information while running', action='store_true')  # on/off flag
parser.add_argument('-t', '--traceback',  help = 'set traceback standard python format (turns off rich formatting of traceback)', action = 'store_true')
args = parser.parse_args()

# setup some terminal formatting and logging
FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level if not args.debug else 'DEBUG', format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
if not args.traceback: install(show_locals=True)

# we can only import these after setting up the logger since src.preprocessing 
# sets up another logger (for secret reasons) which overwrites the log level..
import src.preprocessing as pp
from src.gnn import GCN


###################
### ENTRY POINT ###
###################


# load annotations from gff files and format to pandas dataframe
genome1_name = 'Cav_10DC88'
genome1_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff'))
log.info(f"Loaded annotation file of first genome: {os.path.join('data', 'minimal_Cav_10DC88_RENAMED.gff')}")
log.debug(f"Genome 1 annotation dataframe:\n {genome1_annotation_df}")

genome2_name = 'Cav_11DC096'
genome2_annotation_df = pp.load_gff(os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff'))
log.info(f"Loaded annotation file of second genome: {os.path.join('data', 'minimal_Cav_11DC096_RENAMED.gff')}")
log.debug(f"Genome 2 annotation dataframe:\n {genome2_annotation_df}")

# total number of genes found in all annotation files
num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df.index)
log.info(f"Total number of genes found in annotation files: {num_genes}")

# load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
sim_score_dict = pp.load_similarity_score(os.path.join('data', 'minimal_mmseq2_result.csv'))
log.info(f"Loaded similarity scores file: {os.path.join('data', 'minimal_mmseq2_result.csv')}")

# load holy ribap table to generate labels for test data set
ribap_groups_dict = pp.load_ribap_groups(os.path.join('data', 'holy_python_ribap_95.csv'), [genome1_name, genome2_name])
log.info(f"Loaded RIBAP groups file: {os.path.join('data', 'minimal_mmseq2_result.csv')}")
log.debug(f"Got RIBAP groups dictionary:\n {next(iter(ribap_groups_dict.items()))}")

# low embedding dim will reduce risk of overfitting but may prevent model form learning nuanced patterns
# we later concat the embeddings of each gene ID (node) with its neighbouring gene ID embedding -> 3 * embedding dim
gene_id_embedding_dim = 64

# more dims
hidden_dim = 64

# this (the edge features) later holds the similarity bit score of MMSeqs2 clustering for each two gene nodes connected by an edge
edge_feature_dim = 10

# map string gene IDs to integer IDs and save in tensor, then embed the int IDs in vector with embedding layer
gene_ids_lst = list(genome1_annotation_df.index) + list(genome2_annotation_df.index)
gene_id_integer_dict = {gene: idx for idx, gene in enumerate(gene_ids_lst)}


# reshape tensor to have correct dimensions
gene_ids_ts = torch.tensor(list(gene_id_integer_dict.values()))

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

# construct list of labels from ribap groups and format to match edge_index
labels_ts = pp.map_labels_to_edge_index(edge_index_ts, gene_ids_lst, ribap_groups_dict)
log.info('Created tensor of labels for training from RIBAP groups.')
log.debug(f"Got tensor of labels of shape: {labels_ts.shape}\n{labels_ts}")

# define the edge features (similarity bit scores from MMSeqs2, higher score ~ higher similarity)
edge_weight_ts = pp.map_edge_weights(edge_index_ts, sim_score_dict, gene_ids_lst)#torch.randn((num_genes/2, edge_feature_dim))  # Edge features
batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

# neighbour dictionary with each gene's upstream and downstream neighbours
neighbour_lst = pp.construct_neighbour_lst(len(genome1_annotation_df.index)) + pp.construct_neighbour_lst(len(genome2_annotation_df.index), num_neighbours = args.neighbours)
log.info(f"Constructed neighbours, first entry: {neighbour_lst[0]}; last entry {neighbour_lst[-1]}; length: {len(neighbour_lst)-1}")


# Initialize model, optimizer, and loss function
#model = GeneHomologyGNN(num_genes=num_genes, embedding_dim=gene_id_embedding_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)

dataset = Data(x = gene_ids_ts, edge_index = edge_index_ts, edge_attr = edge_weight_ts, y = labels_ts)
dataset.validate()
log.info(f"Constructed dataset from node, egde and index tensors: {dataset}")


transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(dataset)
log.debug(f"Split dataset into train, test and validation data:\n train: {train_data}\ntest: {test_data}\nvalidation: {val_data}")
#dataset = train_data

# DataLoader expects a list of Data() objects as input smh? or a dataset 
log.info(f"Constructing dataloader with batch size: {args.batch_size}")
dataloader = DataLoader([train_data], batch_size=args.batch_size, shuffle=True)

model = GCN(dataset = dataset, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, neighbour_lst = neighbour_lst)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO: is this a good loss function for our scenario?
criterion = torch.nn.BCELoss()
num_epochs = args.epochs

train_losses = []
train_accuracies = []


if os.path.exists(os.path.join(args.model)):
    log.info(f"Found model file '{args.model}' with trained parameter, restoring model state for inference..")
    model.load_state_dict(torch.load(args.model))
else:
    # Training loop
    log.info(f"Entering training loop with batch size: {args.batch_size}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed: ", total=num_epochs)
        batch_bar    = progress.add_task("Training current batch:", total=len(dataloader))

        for epoch in range(num_epochs):
            correct = 0
            running_loss = 0
            total = 0

            for batch in dataloader:

                model.train()
                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')
                output = model(dataset)
                log.debug('Calling loss function on current batch..')
                log.debug(dataset)
                loss = criterion(output, labels_ts)
                log.debug('Calling backward step on current batch..')
                loss.backward()
                log.debug('Calling optimizer step on current batch..')
                optimizer.step()
                
                # get some metrics, maybe do this in the model class?
                log.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
                #running_loss += loss.item() * gene_ids_ts.size(0)
                _, predicted = torch.max(output, 0)
                total += labels_ts.size(0)
                correct += (predicted == labels_ts).sum().item()

                progress.update(batch_bar, advance = 1)


            epoch_accuracy = 100 * correct / total

            train_losses.append(loss.item())
            train_accuracies.append(epoch_accuracy)
            progress.update(training_bar, advance = 1)


    log.info(f"Finished model training.\nPlotting metrics..")
    log.debug(f"\nLoss: {train_losses}\nAccuracy: {train_accuracies}")
    plot_loss_accuracy(num_epochs, train_losses, train_accuracies)
    log.info(f"Saving model to file '{args.model}'")
    torch.save(model.state_dict(), args.model)

# Prediction example
model.eval()
with torch.no_grad():
    with Console().status("Infering model on test data..") as status:
        pred = model(test_data)
    
    accuracy = (pred == test_data.y).sum().item() / len(test_data.y)
    log.info(f"Correctly predicted: {(pred == test_data.y).sum().item() } out of {len(test_data.y)} genes to be homologs.")
    log.info(f"Accuracy on test dataset: {accuracy}")
    log.info('Exiting.')
