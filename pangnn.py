from src.plot import plot_loss_accuracy, plot_graph
from src.setup import log, args
import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from src.predict import predict_homolog_genes
from rich.progress import Progress
import src.preprocessing as pp
from src.gnn import GCN


###################
### ENTRY POINT ###
###################


# load annotations from gff files and format to pandas dataframe
genome1_name = args.annotation.split(',')[0]
genome1_annotation_df = pp.load_gff(genome1_name)
log.info(f"Loaded annotation file of first genome: {genome1_name}")
log.debug(f"Genome 1 annotation dataframe:\n {genome1_annotation_df}")

genome2_name = args.annotation.split(',')[1]
genome2_annotation_df = pp.load_gff(genome2_name)
log.info(f"Loaded annotation file of second genome: {genome2_name}")
log.debug(f"Genome 2 annotation dataframe:\n {genome2_annotation_df}")

# total number of genes found in all annotation files
num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df.index)
log.info(f"Total number of genes found in annotation files: {num_genes}")

# load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
sim_score_dict = pp.load_similarity_score(args.similarity)
log.info(f"Loaded similarity scores file: {args.similarity}")

# low embedding dim will reduce risk of overfitting but may prevent model form learning nuanced patterns
# we later concat the embeddings of each gene ID (node) with its neighbouring gene ID embedding -> 3 * embedding dim
gene_id_embedding_dim = 64

# more dims
hidden_dim = 64

# this (the edge features) later holds the similarity bit score of MMSeqs2 clustering for each two gene nodes connected by an edge
edge_feature_dim = 10

# map string gene IDs to integer IDs and save in tensor, then embed the int IDs in vector with embedding layer
gene_ids_lst = list(genome1_annotation_df.index) + list(genome2_annotation_df.index)
gene_ids_lst_train = list(genome1_annotation_df.index)[:int(len(genome1_annotation_df.index) * 0.7)] + list(genome2_annotation_df.index)[:int(len(genome2_annotation_df.index) * 0.7)]
gene_ids_lst_test  = list(genome1_annotation_df.index) + list(genome2_annotation_df.index)
gene_id_integer_dict = {gene: idx for idx, gene in enumerate(gene_ids_lst)}


# reshape tensor to have correct dimensions
gene_ids_ts = torch.tensor(list(gene_id_integer_dict.values()))
log.debug(gene_ids_ts)

# index specifying which nodes are connected by an edge in the graph, e.g.:
# edge_index = torch.tensor([[0, 1],  # source nodes
#                           [1, 2]],  # destination nodes
# dtype=torch.long)  # Edge connections
# however since we have an all-to-all (fully) connected graph where every node (gene)
# has a sequence similarity score to every other node in the graph we define the
# fully connected edge index as follows (tensor magic happening):
row = torch.arange(num_genes).repeat(num_genes)
col = row.view(num_genes, num_genes).t().flatten()
mask = (row != col)
# remove self loops
#edge_index_ts = torch.stack((row[mask], col[mask]), dim=0)
edge_index_ts = torch.stack((row, col), dim=0)
log.info(f"Edge index for fully connected graph successfully created.")
log.debug(edge_index_ts)

# index of the element in the dict luckily), then for 
#torch.set_printoptions(threshold=10_000) # set print limit for tensors
#torch.set_printoptions(profile="full")


# define the edge features (similarity bit scores from MMSeqs2, higher score ~ higher similarity)
edge_weight_ts = pp.map_edge_weights(edge_index_ts, sim_score_dict, gene_ids_lst)#torch.randn((num_genes/2, edge_feature_dim))  # Edge features
log.debug(edge_weight_ts)

#batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

# neighbour dictionary with each gene's upstream and downstream neighbours
neighbour_lst = pp.construct_neighbour_lst(len(genome1_annotation_df.index)) + pp.construct_neighbour_lst(len(genome2_annotation_df.index), num_neighbours = args.neighbours)
log.info(f"Constructed neighbours, first entry: {neighbour_lst[0]}; last entry {neighbour_lst[-1]}; length: {len(neighbour_lst)-1}")

# Initialize model, optimizer, and loss function
#model = GeneHomologyGNN(num_genes=num_genes, embedding_dim=gene_id_embedding_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)

if args.train:
    # load holy ribap table to generate labels for test data set
    ribap_groups_dict = pp.load_ribap_groups(args.ribap_groups, [os.path.basename(genome1_name).split('.')[0].replace('_RENAMED', ''), os.path.basename(genome2_name).split('.')[0].replace('_RENAMED', '')])
    log.info(f"Loaded RIBAP groups file: {args.ribap_groups}")
    log.debug(f"Got RIBAP groups dictionary:\n {next(iter(ribap_groups_dict.items()))}")

    # construct list of labels from ribap groups and format to match edge_index
    labels_ts = pp.map_labels_to_edge_index(edge_index_ts, gene_ids_lst, ribap_groups_dict)
    log.info('Created tensor of labels for training from RIBAP groups.')
    log.debug(f"Got tensor of labels of shape: {labels_ts.shape}\n{labels_ts}")
    dataset = Data(x = gene_ids_ts, edge_index = edge_index_ts, edge_attr = edge_weight_ts, y = labels_ts)
else:
    dataset = Data(x = gene_ids_ts, edge_index = edge_index_ts, edge_attr = edge_weight_ts)



if args.plot_graph: plot_graph(dataset, gene_ids_lst, os.path.join('plots', 'input_graph.png'))

dataset.validate()
log.info(f"Constructed dataset from node, egde and index tensors: {dataset}")


transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(dataset)
log.debug(f"Split dataset into train, test and validation data:\n train: {train_data}\ntest: {test_data}\nvalidation: {val_data}")
#dataset = train_data

# DataLoader expects a list of Data() objects as input smh? or a dataset 
args.batch_size = 1
log.info(f"Constructing dataloader with batch size: {args.batch_size}")
dataloader = DataLoader([train_data], batch_size=args.batch_size, shuffle=True)

model = GCN(dataset = dataset, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, neighbour_lst = neighbour_lst)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO: is this a good loss function for our scenario?
# criterion = torch.nn.BCELoss() # if your model outputs probabilities 
criterion = torch.nn.BCEWithLogitsLoss() # if your model outputs raw logits and you want the loss function to handle the sigmoid activation internally

train_losses = []
train_accuracies = []


if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameter, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args))
        predict_homolog_genes(model, dataset)

    else:
        log.error(f"Could not infer model because model parameters file '{args.model_args}' was not found, exiting.")
        quit()
elif args.train:
    # Training loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")
    dataset.to(device)
    model.to(device)
    log.info(f"Entering training loop with batch size: {args.batch_size}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed: ", total=args.epochs)
        batch_bar    = progress.add_task("Training current batch:", total=len(dataloader))

        for epoch in range(args.epochs):
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
    plot_loss_accuracy(args.epochs, train_losses, train_accuracies)
    log.info(f"Saving model to file '{args.model_args}'")
    torch.save(model.state_dict(), args.model_args)

    # get metrics on test dataset
    prediction_bin, prediction_scores = predict_homolog_genes(model, test_data)
    log.debug(prediction_bin)
    log.debug(labels_ts)
    
