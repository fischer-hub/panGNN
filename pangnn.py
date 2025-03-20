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
from rich.progress import track, Console, Progress


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
        dataset = UnionGraphDataset(args.annotation, args.similarity, args.ribap_groups, split=(0.5, 0.2, 0.1), categorical_nodes = args.categorical_node) if args.train else HomogenousDataset(args.annotation, args.similarity)
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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.gpu else 'cpu'
device = accelerator.device
#model = MyGCN(dataset = dataset.train, hidden_dim = hidden_dim, num_neighbours = args.neighbours, node_feature_dim = gene_id_embedding_dim, device = device)
model = AlternateGCN(device = device, dataset = dataset.train, categorical_nodes = dataset.categorical_nodes, dims = [args.node_dim, args.hidden_dim])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#selflr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = 7, factor = 0.6)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
#model, optimizer, data = accelerator.prepare(model, optimizer, dataset)

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

train_data_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True, pin_memory = True)
val_data_loader = DataLoader(dataset.val, batch_size=args.batch_size, shuffle=False, pin_memory = True)
test_data_loader = DataLoader(dataset.test, batch_size=len(dataset.test), shuffle=False, pin_memory = True)

model, optimizer, train_data_loader, scheduler, val_data_loader, test_data_loader = accelerator.prepare(model, optimizer, train_data_loader, scheduler, val_data_loader, test_data_loader)

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
    #model.to(device)

    log.info(f"Entering training loop with batch size: {args.batch_size}, class balance: {dataset.class_balance}, {len(train_data_loader)} batches.")#{dataset.class_balance}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed:", total=args.epochs)
        batch_bar    = progress.add_task("Training model on training data set", total=len(train_data_loader))
        val_bar      = progress.add_task("Infering model on validation data set:", total=len(val_data_loader))

        for epoch in range(args.epochs):
            total = 0

            # shuffle list of input graphs so the model sees the data in different order every time 
            #random.shuffle(dataset.train)
            accuracy = []
            epoch_correct = 0
            epoch_total = 0
            val_loss, train_loss = 0, 0
            all_labels_val, all_probabilities_val, all_predictions_val = [], [], []
            all_labels_train, all_probabilities_train, all_predictions_train = [], [], []

            for batch_num, batch in enumerate(train_data_loader):
            #for batch_num, batch in enumerate(dataset.train):
            #for batch_num in range(args.num_batches):
                
                model.train()

                #batch = sub_sample_graph_edges(dataset.train, device, fraction = 0.8) if not args.union_edge_weights else dataset.train
                #batch = dataset.train
                #dataset.graph_to(batch, device)

                labels = batch[0].y if isinstance(batch, tuple) else batch.y

                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')

                output = model(batch)
                #print(f"logits after decoding:\n{output}")
                log.debug('Calling loss function on current batch..')
                log.debug(batch)
                loss = criterion(output, labels)
                train_loss += loss
                log.debug('Calling backward step on current batch..')
                #loss.backward()
                accelerator.backward(loss)
                log.debug('Calling optimizer step on current batch..')
                optimizer.step()
                
                probabilities = torch.sigmoid(output).cpu()
                binary_prediction_train = (probabilities >= binary_th).int()
                
                all_probabilities_train += list(probabilities)
                all_predictions_train += list(binary_prediction_train)
                all_labels_train += list(labels.cpu())


                if args.dynamic_binary_threshold:
                    fpr, tpr, thresholds = roc_curve(labels, torch.sigmoid(output.detach()))
                    youden_index = tpr - fpr
                    optimal_threshold = thresholds[youden_index.argmax()]
                    writer.add_scalar("Optimal_Binary_Threshold/val", optimal_threshold, epoch)
                    binary_th = optimal_threshold

                    print(optimal_threshold)
                
                progress.update(batch_bar, advance = 1)

                """                 for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        print(f'{name}: {param.grad.mean()}') """

            with torch.no_grad():  # Disable gradient calculation for validation

                for num_batch, batch in enumerate(val_data_loader):

                    model.eval()
                    output = model(batch)

                    val_labels = batch.y

                    loss = criterion(output, val_labels)
                    scheduler.step(loss)
                    val_loss += loss

                    probabilities = torch.sigmoid(output).cpu()
                    binary_prediction_val = (probabilities >= binary_th).int()
                    
                    val_labels = val_labels.cpu()
                    all_probabilities_val += list(probabilities)
                    all_predictions_val += list(binary_prediction_val.cpu())
                    all_labels_val += list(val_labels)
        
                    progress.update(val_bar, advance = 1)


                
            # val metrics
            conf_matrix = confusion_matrix(all_labels_val, all_predictions_val, labels = [0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()

            fpr, tpr, thresholds = roc_curve(all_labels_val, all_probabilities_val)
            roc_auc_val = auc(fpr, tpr)
            pr_auc_val = average_precision_score(val_labels, probabilities)

            precision_val = tp / (tp + fp + 1e-10)
            recall_val = tp / (tp + fn + 1e-10)
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
            acc_val = (tp + tn) / (tp + tn + fp + fn)

            writer.add_scalar("ROC-AUC/val", roc_auc_val, epoch)
            writer.add_scalar("PR-AUC/val", pr_auc_val, epoch)
            writer.add_scalar("Loss/val", val_loss/len(val_data_loader), epoch)
            writer.add_scalar("Acc/val", acc_val, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("precision/val", precision_val, epoch)
            writer.add_scalar("recall/val", recall_val, epoch)
            writer.add_scalar("F1/val", f1_val, epoch)

            #train metrics
            conf_matrix = confusion_matrix(all_labels_train, all_predictions_train, labels = [0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()

            precision_train = tp / (tp + fp + 1e-10)
            recall_train = tp / (tp + fn + 1e-10)
            f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-10)
            acc_train = (tp + tn) / (tp + tn + fp + fn)

            writer.add_scalar("Loss/train", train_loss/len(train_data_loader), epoch)
            writer.add_scalar("Acc/train", acc_train, epoch)
            writer.add_scalar("F1/train", f1_train, epoch)
        
            # get some metrics, maybe do this in the model class?
            log.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc_train:.4f}, F1 {f1_train:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc_val:.4f}, LR: {optimizer.param_groups[0]['lr']:.10f},  Val F1: {f1_val:.4f}, Val AP: {pr_auc_val:.4f}{f'Optimal Bin. Th. {binary_th:.4f}' if args.dynamic_binary_threshold else ''}")
            
            progress.update(training_bar, advance = 1)
            progress.reset(batch_bar)
            progress.reset(val_bar)

    #plot_loss_accuracy(args.epochs, train_losses, train_accuracies, val_losses, val_accuracies, f1_train_lst)
    log.info('Unwrapping model from accelerate layers.')
    model = accelerator.unwrap_model(model)
    log.info(f"Saving model to file '{os.path.join('temp', run_id, args.model_args)}'")
    torch.save(model.state_dict(), os.path.join('temp', run_id, args.model_args))

    with Console().status("Finished model training, plotting metrics..") as status:
        # get metrics on test dataset
        for batch in test_data_loader:
            prediction_bin, prediction_scores, stats = predict_homolog_genes(model, None, batch, binary_th=binary_th)
            writer.add_pr_curve('PR/val', batch.y.cpu(), torch.sigmoid(prediction_scores))

    writer.add_hparams(hparams, stats)
    log.debug(prediction_bin)
    log.info(f"Time elapsed: {time.time() - start:.4f} seconds")

    stats['binary_threshold'] = binary_th
    stats['date'] = str(datetime.date.today())
    stats['neighbours'] = args.neighbours
    #stats['num_nodes_train'] = len(dataset.train.x)
    #stats['num_nodes_sim_edges_train'] = len(dataset.train.edge_index)
    #stats['num_nodes_test'] = len(dataset.test.x)
    #stats['num_nodes_sim_edges_test'] = len(dataset.test.edge_index)
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