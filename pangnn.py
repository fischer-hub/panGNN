#!/usr/bin/env python3
from src.plot import plot_loss_accuracy, plot_graph, plot_simscore_class, plot_logit_distribution, plot_union_graph, plot_simscore_distribution_by_class
from src.setup import log, args
import torch, os, random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from src.predict import predict_homolog_genes
from rich.progress import Progress
from src.dataset import HomogenousDataset, UnionGraphDataset
from src.gnn import MyGCN, AlternateGCN
from src.simulate import generate_data
from sklearn.metrics import confusion_matrix
from src.postprocessing import write_groups_file
from src.helper import generate_minimal_dataset, simulate_dataset
from sklearn.metrics import roc_curve

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


#dataset = HomogenousDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)
#dataset = UnionGraphDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours, split=(0.98, 0.02)) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)
#dataset.generate_graph_data()
dataset  = UnionGraphDataset()
dataset.simulate_dataset(50000, 25, 0.15)
print(dataset.train.edge_attr.max())
plot_simscore_distribution_by_class(dataset.train, path= os.path.join('plots', 'sim_score_distribution_by_class_simulated.png'))

#dataset = generate_minimal_dataset()
#dataset.train = generate_minimal_dataset()
#dataset.test = generate_minimal_dataset()

#plot_union_graph(dataset, os.path.join('plots', 'union_graph.png'))


#dataset.split_data((0.8,0.20,0), batch_size = args.batch_size)

log.info(f"Constructed train dataset from node, egde and index tensors: {dataset.train}")

#plot_logit_distribution(dataset.train.edge_weight_ts, path= os.path.join('plots', 'sim_score_distribution_unscaled.png'))

#if args.plot_graph: plot_graph(dataset.train, os.path.join('plots', 'input_graph.png'))

#plot_simscore_class(dataset.train)

log.info(f"Constructed test dataset from node, egde and index tensors: {dataset.test}")
#log.info(f"Constructed {dataset.train[1].x}")
#log.info(f"Constructed {dataset.train[1].edge_index}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.gpu else 'cpu'
#model = MyGCN(dataset = dataset.train, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, device = device)
model = AlternateGCN(device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#lr=0.00005)

# TODO: is this a good loss function for our scenario?
# criterion = torch.nn.BCELoss() # if your model outputs probabilities, outputs logits
# nn.CrossEntropyLoss() # multi-class classification where each sample belongs to only one class out of multiple classes., outputs logits
criterion = torch.nn.BCEWithLogitsLoss()#pos_weight = dataset.class_balance) # if your model outputs raw logits and you want the loss function to handle the sigmoid activation internally, outputs probabilities

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
f1_train_lst = []
precision_lst = []
recall_lst = []
binary_th = args.binary_threshold # 0.53

if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameter, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args))
        prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train, dataset.test, binary_th = binary_th)

    else:
        log.error(f"Could not infer model because model parameters file '{args.model_args}' was not found, exiting.")
        quit()

elif args.train:
    # Training loop
    log.info(f"Training on device: {device}")
    #dataset.train.to(device)
    #model.to(device)
    log.info(f"Entering training loop with: {args.num_batches} batches, class weight ")#{dataset.class_balance}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed:", total=args.epochs)
        batch_bar    = progress.add_task("Training current batch:", total=args.num_batches)

        for epoch in range(args.epochs):
            total = 0

            # shuffle list of input graphs so the model sees the data in different order every time 
            #random.shuffle(dataset.train)
            accuracy = []
            epoch_correct = 0
            epoch_total = 0
            val_loss = 0

            #for batch_num, batch in enumerate(dataset.train):
            for batch_num in range(args.num_batches):

                batch = dataset.sub_sample_graph_edges(dataset.train, fraction = 0.8)
                #batch = dataset.train
                labels = batch[0].y if isinstance(batch, tuple) else batch.y
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight =(labels == 0.).sum()/labels.sum())

                model.train()
                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')
                output = model(batch)
                log.debug('Calling loss function on current batch..')
                log.debug(batch)
                loss = criterion(output, labels)
                log.debug('Calling backward step on current batch..')
                loss.backward()
                log.debug('Calling optimizer step on current batch..')
                optimizer.step()

                progress.update(batch_bar, advance = 1)

                if args.dynamic_binary_threshold:
                    fpr, tpr, thresholds = roc_curve(labels, output)
                    youden_index = tpr - fpr
                    optimal_threshold = thresholds[youden_index.argmax()]
                    binary_th = optimal_threshold
                
                binary_prediction = (torch.sigmoid(output) >= binary_th).int()
                accuracy.append(((binary_prediction == labels).sum().item()) / len(labels))

                epoch_correct += (binary_prediction == labels).sum().item()  # Count correct predictions
                epoch_total += len(labels)  # Total number of samples in the batch
                batch_accuracy = (binary_prediction == labels).sum().item() / len(labels)  # Calculate accuracy

                """                 for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        print(f'{name}: {param.grad.mean()}') """

            with torch.no_grad():  # Disable gradient calculation for validation
                model.eval()
                output = model(dataset.test)

                test_labels = dataset.test[0].y if isinstance(dataset.test, tuple) else dataset.test.y

                val_loss = criterion(output, test_labels)
                binary_prediction_val = (torch.sigmoid(output) >= binary_th).int()
                val_acc = (binary_prediction_val == test_labels).sum().item() / len(test_labels)
                
                conf_matrix = confusion_matrix(test_labels, binary_prediction_val)
                tn, fp, fn, tp = conf_matrix.ravel()
                f1_val = (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

                conf_matrix = confusion_matrix(labels, binary_prediction)
                tn, fp, fn, tp = conf_matrix.ravel()
                f1_train = (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))

                precision_lst.append(tp/(tp+fp))
                recall_lst.append(tp/(tp+fn))
                #f1_val = 0 
                #f1_train = 0

        
            # get some metrics, maybe do this in the model class?
            log.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {epoch_correct / epoch_total:.4f}, F1 {f1_train:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f},  Val F1: {f1_val:.4f}{f'Optimal Bin. Th. {binary_th:.4f}' if args.dynamic_binary_threshold else ''}")

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            train_accuracies.append(epoch_correct / epoch_total)
            val_accuracies.append(val_acc)

            f1_train_lst.append(f1_train)
            
            progress.update(training_bar, advance = 1)
            progress.reset(batch_bar)

    log.info(f"Finished model training.\nPlotting metrics..")
    log.debug(f"\nLoss: {train_losses}\nAccuracy: {train_accuracies}")
    plot_loss_accuracy(args.epochs, train_losses, train_accuracies, val_losses, val_accuracies, f1_train_lst)
    log.info(f"Saving model to file '{args.model_args}'")
    torch.save(model.state_dict(), args.model_args)

    # get metrics on test dataset
    prediction_bin, prediction_scores = predict_homolog_genes(model, dataset.train, dataset.test, binary_th=binary_th)
    log.debug(prediction_bin)
    
#write_groups_file(dataset.test, prediction_bin)