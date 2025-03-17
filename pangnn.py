#!/usr/bin/env python3
from src.plot import plot_loss_accuracy, plot_graph, plot_simscore_class, plot_logit_distribution, plot_union_graph, plot_simscore_distribution_by_class, plot_umap_pca
from src.setup import log, args, hparams
import torch, os, random, datetime, time, shutil, resource
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from src.predict import predict_homolog_genes
from rich.progress import Progress
from src.dataset import HomogenousDataset, UnionGraphDataset
from src.gnn import MyGCN, AlternateGCN
from src.simulate import generate_data
from sklearn.metrics import confusion_matrix
from src.postprocessing import write_groups_file, write_stats_csv
from src.helper import generate_minimal_dataset, simulate_dataset, sub_sample_graph_edges
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

accelerator = Accelerator()
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_limit < 50000: resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit-10, hard_limit))


#dataset = HomogenousDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)
if not args.simulate_dataset:
    if args.from_pickle:
        dataset  = UnionGraphDataset()
        dataset.load(args.from_pickle)
        num_genomes = 'number of genomes not available since dataset was loaded from disk'
    else:
        num_genomes = len(args.annotation)
        dataset = UnionGraphDataset(args.annotation, args.similarity, args.ribap_groups, split=(0.6, 0.4), categorical_nodes = args.categorical_node) if args.train else HomogenousDataset(args.annotation, args.similarity)
    #dataset.generate_graph_data()
else:
    log.info('Simulating dataset.')
    num_genomes = 3
    dataset  = UnionGraphDataset()
    dataset.simulate_dataset(2000, num_genomes, 0.15)

hparams['num_genomes'] = num_genomes
hparams['num_genes'] = dataset.num_genes
hparams['class_balance'] = dataset.class_balance

#print(dataset.train.edge_attr.max())
#plot_simscore_distribution_by_class(dataset.train, path= os.path.join('plots', 'sim_dist', '50_genomes_sim_score_distribution_by_class.png'))
#dataset = generate_minimal_dataset()
#dataset.train = generate_minimal_dataset()
#dataset.test = generate_minimal_dataset()
#plot_union_graph(dataset, os.path.join('plots', 'union_graph.png'))


#dataset.split_data((0.8,0.20,0), batch_size = args.batch_size)

log.debug(f"Constructed train dataset from node, egde and index tensors: {dataset.train}")
#plot_logit_distribution(dataset.train.edge_attr, path= os.path.join('plots', 'normalized_sim_scores_prob.png'))
#if args.plot_graph: plot_graph(dataset.train, os.path.join('plots', 'input_graph.png'))

#plot_simscore_class(dataset.train)

log.debug(f"Constructed test dataset from node, egde and index tensors: {dataset.test}")
#log.info(f"Constructed {dataset.train[1].x}")
#log.info(f"Constructed {dataset.train[1].edge_index}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.gpu else 'cpu'
#model = MyGCN(dataset = dataset.train, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, device = device)
model = AlternateGCN(device = device, dataset = dataset.train, categorical_nodes = dataset.categorical_nodes, dims = [args.node_dim, args.hidden_dim])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#selflr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = 7, factor = 0.6)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
model, optimizer, data = accelerator.prepare(model, optimizer, dataset)

# TODO: is this a good loss function for our scenario?
# criterion = torch.nn.BCELoss() # if your model outputs probabilities, outputs logits
# nn.CrossEntropyLoss() # multi-class classification where each sample belongs to only one class out of multiple classes., outputs logits
#criterion = torch.nn.BCEWithLogitsLoss()#pos_weight = dataset.class_balance) # if your model outputs raw logits and you want the loss function to handle the sigmoid activation internally, outputs probabilities

criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(hparams['class_balance']))

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
f1_train_lst = []
precision_lst = []
recall_lst = []
binary_th = args.binary_threshold

start = time.time()

if os.path.exists('temp'):
    log.info('Clearing temporary directory from previous runs')
    shutil.rmtree("temp")
    os.mkdir('temp')
else:
    os.mkdir('temp')

if not os.path.exists('runs'): os.mkdir('runs')

run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + args.tb_comment
writer = SummaryWriter(log_dir = os.path.join('temp', run_id), comment = args.tb_comment)

if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameter, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args))
        prediction_bin, prediction_scores, stats = predict_homolog_genes(model, dataset.train, dataset.test, binary_th = binary_th)
        stats['mode'] = 'test'

    else:
        log.error(f"Could not infer model because model parameters file '{args.model_args}' was not found, exiting.")
        quit()

elif args.train:
    # Training loop
    log.info(f"Training on device: {device}")

    #dataset.to(device)
    model.to(device)

    log.info(f"Entering training loop with batch size: {args.batch_size}, class balance: {dataset.class_balance}, {len(dataset.train)} batches.")#{dataset.class_balance}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed:", total=args.epochs)
        batch_bar    = progress.add_task("Training current batch set:", total=len(dataset.train))

        for epoch in range(args.epochs):
            total = 0

            # shuffle list of input graphs so the model sees the data in different order every time 
            random.shuffle(dataset.train)
            accuracy = []
            epoch_correct = 0
            epoch_total = 0
            val_loss = 0

            for batch_num, batch in enumerate(dataset.train):
            #for batch_num in range(args.num_batches):
                
                model.train()

                #batch = sub_sample_graph_edges(dataset.train, device, fraction = 0.8) if not args.union_edge_weights else dataset.train
                #batch = dataset.train
                #dataset.graph_to(batch, device)

                #batch = dataset.train
                #print(batch)
                labels = batch[0].y if isinstance(batch, tuple) else batch.y
                #class_balance_factor = (labels == 0.).sum()/labels.sum()
                #criterion = torch.nn.BCEWithLogitsLoss(pos_weight = min(10, max(0.1, class_balance_factor)))
                #print(class_balance_factor)

                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')
                output = model(batch)
                #print(f"logits after decoding:\n{output}")
                log.debug('Calling loss function on current batch..')
                log.debug(batch)
                loss = criterion(output, labels)
                log.debug('Calling backward step on current batch..')
                loss.backward()
                log.debug('Calling optimizer step on current batch..')
                optimizer.step()

                progress.update(batch_bar, advance = 1)

                if args.dynamic_binary_threshold:
                    fpr, tpr, thresholds = roc_curve(labels, torch.sigmoid(output.detach()))
                    youden_index = tpr - fpr
                    optimal_threshold = thresholds[youden_index.argmax()]
                    writer.add_scalar("Optimal_Binary_Threshold/val", optimal_threshold, epoch)
                    binary_th = optimal_threshold

                    print(optimal_threshold)
                
                binary_prediction = (torch.sigmoid(output) >= binary_th).int()
                accuracy.append(((binary_prediction == labels).sum().item()) / len(labels))

                epoch_correct += (binary_prediction == labels).sum().item()  # Count correct predictions
                epoch_total += len(labels)  # Total number of samples in the batch
                batch_accuracy = (binary_prediction == labels).sum().item() / len(labels)  # Calculate accuracy
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Acc/train", ((binary_prediction == labels).sum().item()) / len(labels), epoch)
                """                 for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        print(f'{name}: {param.grad.mean()}') """

            with torch.no_grad():  # Disable gradient calculation for validation
                model.eval()
                dataset.test.to(device)
                output = model(dataset.test)

                test_labels = dataset.test[0].y if isinstance(dataset.test, tuple) else dataset.test.y

                val_loss = criterion(output, test_labels)
                writer.add_scalar("Loss/val", val_loss, epoch)
                #scheduler.step()
                scheduler.step(val_loss)
                probabilities = torch.sigmoid(output)
                binary_prediction_val = (probabilities >= binary_th).int()
                val_acc = (binary_prediction_val == test_labels).sum().item() / len(test_labels)
                writer.add_scalar("Acc/val", val_acc, epoch)
                
                test_labels = test_labels.cpu()
                binary_prediction_val = binary_prediction_val.cpu()
                binary_prediction = binary_prediction.cpu()
                labels = labels.cpu()
                conf_matrix = confusion_matrix(test_labels, binary_prediction_val)
                tn, fp, fn, tp = conf_matrix.ravel()

                fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
                roc_auc = auc(fpr, tpr)
                average_precision = average_precision_score(test_labels, probabilities)

                writer.add_scalar("ROC_AUC/val", roc_auc, epoch)
                writer.add_scalar("AP/val", average_precision, epoch)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
                
                precision_val = tp / (tp + fp + 1e-10)
                recall_val = tp / (tp + fn + 1e-10)
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
                #f1_val = (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))
                writer.add_scalar("F1/val", f1_val, epoch)


                conf_matrix = confusion_matrix(labels, binary_prediction, labels = [0, 1])
                tn, fp, fn, tp = conf_matrix.ravel()
                precision_train = tp / (tp + fp + 1e-10)
                recall_train = tp / (tp + fn + 1e-10)
                f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-10)

                precision_lst.append(precision_train)
                writer.add_scalar("precision/val", precision_val, epoch)

                recall_lst.append(recall_val)
                writer.add_scalar("recall/val", recall_val, epoch)
                #f1_val = 0 
                #f1_train = 0

        
            # get some metrics, maybe do this in the model class?
            log.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {epoch_correct / epoch_total:.4f}, F1 {f1_train:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.10f},  Val F1: {f1_val:.4f}, Val AP: {average_precision:.4f}{f'Optimal Bin. Th. {binary_th:.4f}' if args.dynamic_binary_threshold else ''}")

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
    log.info(f"Saving model to file '{os.path.join('temp', run_id, args.model_args)}'")
    torch.save(model.state_dict(), os.path.join('temp', run_id, args.model_args))

    # get metrics on test dataset
    prediction_bin, prediction_scores, stats = predict_homolog_genes(model, dataset.train, dataset.test, binary_th=binary_th)
    print(hparams)
    writer.add_pr_curve('PR/val', dataset.test.y, torch.sigmoid(prediction_scores))
    writer.add_hparams(hparams, stats)
    log.debug(prediction_bin)
    log.info(f"Time elapsed: {time.time() - start:.4f} seconds")

    stats['binary_threshold'] = binary_th
    stats['date'] = str(datetime.date.today())
    stats['neighbours'] = args.neighbours
    #stats['num_nodes_train'] = len(dataset.train.x)
    #stats['num_nodes_sim_edges_train'] = len(dataset.train.edge_index)
    stats['num_nodes_test'] = len(dataset.test.x)
    stats['num_nodes_sim_edges_test'] = len(dataset.test.edge_index)
    stats['mode'] = 'train'
    stats['epochs'] = args.epochs
    stats['batch_size'] = args.batch_size
    stats['device'] = device
    stats['runtime'] = (time.time() - start)
    stats['num_genomes'] = num_genomes
    write_stats_csv(stats)
    writer.flush()
    

writer.close()
shutil.move(os.path.join('plots', 'pr_curve.png'), os.path.join('temp', run_id, run_id + 'pr_curve.png'))
#dataset.save(os.path.join('temp', run_id, run_id + '_dataset.pickle'))
shutil.move(os.path.join('temp', run_id), 'runs')
#write_groups_file(dataset.test, prediction_bin)
# map quality Q score transform
# test different architectures / scores on sim data, then introduce hard cases in sim data and 
# test again, 8-20 genomes
# shuffle gene synteny (by block), hardest case - just shuffle all gibs sampling
# compare metrics, auc, pr, f1 etc..
# improvement?