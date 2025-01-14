#!/usr/bin/env python3
from src.plot import plot_loss_accuracy, plot_graph, plot_simscore_class, plot_logit_distribution
from src.setup import log, args
import torch, os, random
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

dataset = HomogenousDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)

dataset.generate_graph_data()
dataset.split_data(batch_size = args.batch_size)

""" import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
umap_model = umap.UMAP(n_components=3, random_state=42)
dist = []
for origin_node, target_node in zip(dataset.test.edge_index[0], dataset.test.edge_index[1]):
    dist.append(abs(dataset.test.x[origin_node] - dataset.test.x[target_node]) * 10000)

#log.info(len(dataset.test.edge_attr), len(dist))
edge_features = np.column_stack((dataset.test.edge_attr, dist))  # Shape: [num_edges, 2]
edge_embedding_2d = umap_model.fit_transform(edge_features)
#pca = PCA(n_components=2)  # Reduce to 2 dimensions
#edge_embedding_2d = umap.fit_transform(edge_features)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(edge_embedding_2d[:, 0], edge_embedding_2d[:, 1], edge_embedding_2d[:, 2], 
                     c=dataset.test.y, cmap='Spectral', s=5)
ax.set_title("3D UMAP Projection of Edge Features")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")

fig.colorbar(scatter, label="Edge Labels")
plt.show()
quit() """

log.info(f"Constructed train dataset from node, egde and index tensors: {dataset.train[0]}")

#plot_logit_distribution(dataset.train.edge_weight_ts, path= os.path.join('plots', 'sim_score_distribution_unscaled.png'))
#dataset.scale_weights()
#dataset.train.concate_edge_weights()
#dataset.split_data()

#plot_logit_distribution(dataset.train.edge_weight_ts, path= os.path.join('plots', 'sim_score_distribution_scaled.png'))

#if args.plot_graph: plot_graph(dataset.train, os.path.join('plots', 'input_graph.png'))

#plot_simscore_class(dataset.train)

log.info(f"Constructed test dataset from node, egde and index tensors: {dataset.test}")
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

if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameter, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args))
        prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train, dataset.test)

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
        batch_bar    = progress.add_task("Training current batch:", total=len(dataset.train))

        for epoch in range(args.epochs):
            total = 0

            # shuffle list of input graphs so the model sees the data in different order every time 
            random.shuffle(dataset.train)
            accuracy = []

            for batch_num, batch in enumerate(dataset.train):

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

                progress.update(batch_bar, advance = 1)
                
                binary_prediction = (torch.sigmoid(output) >= 0.5).int()
                accuracy.append(((binary_prediction == batch.y).sum().item()) / len(batch.y))

            # get some metrics, maybe do this in the model class?
            log.info(f'Epoch {epoch+1}, Loss: {loss.item()}, Acc: {sum(accuracy) / len(accuracy)}')

            train_losses.append(loss.item())
            train_accuracies.append(sum(accuracy) / len(accuracy))
            
            progress.update(training_bar, advance = 1)
            progress.reset(batch_bar)


    log.info(f"Finished model training.\nPlotting metrics..")
    log.debug(f"\nLoss: {train_losses}\nAccuracy: {train_accuracies}")
    plot_loss_accuracy(args.epochs, train_losses, train_accuracies)
    log.info(f"Saving model to file '{args.model_args}'")
    torch.save(model.state_dict(), args.model_args)

    # get metrics on test dataset
    prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train, dataset.test)
    log.debug(prediction_bin)
    
#write_groups_file(dataset.test, prediction_bin)