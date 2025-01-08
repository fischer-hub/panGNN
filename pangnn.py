#!/usr/bin/env python3
from src.plot import plot_loss_accuracy, plot_graph, plot_simscore_class, plot_logit_distribution
from src.setup import log, args
import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from src.predict import predict_homolog_genes
from rich.progress import Progress
from src.dataset import HomogenousDataset
from src.gnn import MyGCN
from src.simulate import generate_data
from src.postprocessing import write_groups_file

###################
### ENTRY POINT ###
###################



# low embedding dim will reduce risk of overfitting but may prevent model form learning nuanced patterns
# we later concat the embeddings of each gene ID (node) with its neighbouring gene ID embedding -> 3 * embedding dim
gene_id_embedding_dim = 64

# more dims
hidden_dim = 64

# this (the edge features) later holds the similarity bit score of MMSeqs2 clustering for each two gene nodes connected by an edge
edge_feature_dim = 128

#batch = torch.zeros(num_genes, dtype=torch.long)  # Batch vector for mini-batches if needed

#simuoated_dataset = generate_data(20000, 5, 5, 0.5, 0.5, 0.02)
class MyObject:
    def __init__(self, attribute1, attribute2):
        self.test = attribute1
        self.train = attribute2

dataset = MyObject
dataset.train = HomogenousDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)
dataset.test = HomogenousDataset(['data/Cga_08-1274-3_RENAMED.gff', 'data/Cga_12-4358_RENAMED.gff'], args.similarity, args.ribap_groups, args.neighbours)
dataset.train.generate_graph_data()
dataset.test.generate_graph_data()
#dataset.train.scale_weights()
#dataset.test.scale_weights()
log.info(f"Constructed dataset from node, egde and index tensors: {dataset.train.data_lst}")

#plot_logit_distribution(dataset.train.edge_weight_ts, path= os.path.join('plots', 'sim_score_distribution_unscaled.png'))
#dataset.scale_weights()
#dataset.train.concate_edge_weights()
#dataset.split_data()

#plot_logit_distribution(dataset.train.edge_weight_ts, path= os.path.join('plots', 'sim_score_distribution_scaled.png'))

#if args.plot_graph: plot_graph(dataset.train, os.path.join('plots', 'input_graph.png'))

#plot_simscore_class(dataset.train)

log.info(f"Constructed dataset from node, egde and index tensors: {dataset.train.data_lst}")
#log.info(f"Constructed {dataset.train[1].x}")
#log.info(f"Constructed {dataset.train[1].edge_index}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.gpu else 'cpu'
model = MyGCN(dataset = dataset.train, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO: is this a good loss function for our scenario?
# criterion = torch.nn.BCELoss() # if your model outputs probabilities, outputs logits
# nn.CrossEntropyLoss() # multi-class classification where each sample belongs to only one class out of multiple classes., outputs logits
criterion = torch.nn.BCEWithLogitsLoss() # if your model outputs raw logits and you want the loss function to handle the sigmoid activation internally, outputs probabilities

train_losses = []
train_accuracies = []
log.info(f"edge weights sum : {dataset.train.edge_weight_ts.sum()}")

if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameter, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args))
        prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train.data_lst[0], dataset.test.data_lst[0])

    else:
        log.error(f"Could not infer model because model parameters file '{args.model_args}' was not found, exiting.")
        quit()

elif args.train:
    # Training loop
    log.info(f"Training on device: {device}")
    #dataset.train.to(device)
    #model.to(device)
    log.info(f"Entering training loop with batch size: {args.batch_size}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed:", total=args.epochs)
        #batch_bar    = progress.add_task("Training current batch:", total=len(dataloader))

        for epoch in range(args.epochs):
            total = 0

            for batch in dataset.train:

                model.train()
                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')
                output = model(batch)
                log.debug('Calling loss function on current batch..')
                log.debug(batch)
                loss = criterion(output, batch.y)
                log.debug('Calling backward step on current batch..')
                loss.backward()
                log.debug('Calling optimizer step on current batch..')
                optimizer.step()
                
                binary_prediction = torch.tensor((torch.sigmoid(output) >= 0.5).int())
                accuracy = ((binary_prediction == batch.y).sum().item()) / len(batch.y)

                # get some metrics, maybe do this in the model class?
                log.info(f'Epoch {epoch+1}, Loss: {loss.item()}, Acc: {accuracy}')

                train_losses.append(loss.item())
                train_accuracies.append(accuracy)
                progress.update(training_bar, advance = 1)


    log.info(f"Finished model training.\nPlotting metrics..")
    log.debug(f"\nLoss: {train_losses}\nAccuracy: {train_accuracies}")
    plot_loss_accuracy(args.epochs, train_losses, train_accuracies)
    log.info(f"Saving model to file '{args.model_args}'")
    torch.save(model.state_dict(), args.model_args)

    # get metrics on test dataset
    prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train.data_lst[0], dataset.test.data_lst[0])
    log.debug(prediction_bin)
    
#write_groups_file(dataset.test, prediction_bin)