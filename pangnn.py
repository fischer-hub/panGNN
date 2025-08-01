#!/usr/bin/env python3
#from src.plot import plot_loss_accuracy, plot_graph, plot_simscore_class, plot_logit_distribution, plot_union_graph, plot_simscore_distribution_by_class, plot_umap_pca
from src.setup import log, args, hparams
import torch, os, datetime, time, shutil, resource, cProfile, pstats, sys
from torch_geometric.loader import DataLoader
from src.predict import predict_homolog_genes
from rich.progress import Progress
from src.dataset import HomogenousDataset, UnionGraphDataset
from src.gnn import AlternateGCN
from src.postprocessing import write_stats_csv
from sklearn.metrics import roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAveragePrecision, BinaryAUROC
from rich.progress import Console, Progress
from src.helper import calculate_logit_baseline_labels, format_duration

#profiler = cProfile.Profile()
#profiler.enable()

# make tensorflow shut up because we only use it for tensorboard and it keeps warning about gpu issues we dont care abpout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

accelerator = Accelerator(mixed_precision = args.mixed_precision)

binary_confusion_matrix_train = BinaryConfusionMatrix().to(accelerator.device)
binary_confusion_matrix_val = BinaryConfusionMatrix().to(accelerator.device)
binary_auroc = BinaryAUROC().to(accelerator.device)
binary_average_precision = BinaryAveragePrecision().to(accelerator.device)

# if limit is too low this can crash the multithreading in preprocessing sadly
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_limit < 50000: resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit-10, hard_limit))

log.info(f"Launched panGNN with command: {' '.join(sys.argv)}")

#dataset = HomogenousDataset(args.annotation, args.similarity, args.ribap_groups, args.neighbours) if args.train else HomogenousDataset(args.annotation, args.similarity, args.neighbours)
if not args.simulate_dataset:
    if args.from_pickle and not args.fix_dataset:
        dataset  = UnionGraphDataset()
        dataset.load(args.from_pickle)
        num_genomes = 'number of genomes not available since dataset was loaded from disk'
    else:
        num_genomes = len(args.annotation)
        dataset = UnionGraphDataset(args.annotation, args.similarity, args.ribap_groups, split=(0.7, 0.15, 0.01), categorical_nodes = args.categorical_node, calculate_baseline=True)
        
        if args.from_pickle and args.fix_dataset:
            dataset.load(args.from_pickle)
        elif args.fix_dataset and not args.from_pickle:
            log.error("Fix dataset was set but no pickle file to load dataset to fix was defined. Please define via '--from_pickle'.")
            quit()
else:
    log.info('Simulating dataset.')
    dataset = UnionGraphDataset(calculate_baseline = True, split=(0.7, 0.15, 0.01), categorical_nodes = args.categorical_node)
    num_genomes = args.simulate_dataset[1]
    #dataset  = UnionGraphDataset()
    #dataset.simulate_dataset(2000, num_genomes, 0.15)

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
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor = 0.6)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
#model, optimizer, data = accelerator.prepare(model, optimizer, dataset)

# TODO: is this a good loss function for our scenario?
# criterion = torch.nn.BCELoss() # if your model outputs probabilities, outputs logits
# nn.CrossEntropyLoss() # multi-class classification where each sample belongs to only one class out of multiple classes., outputs logits
#criterion = torch.nn.BCEWithLogitsLoss()#pos_weight = dataset.class_balance) # if your model outputs raw logits and you want the loss function to handle the sigmoid activation internally, outputs probabilities

criterion = torch.nn.BCEWithLogitsLoss(pos_weight = hparams['class_balance'])

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
f1_train_lst = []
precision_lst = []
recall_lst = []
binary_th = args.binary_threshold
oom_count = 0

start = time.time()

if os.path.exists('temp'):
    log.info('Clearing temporary directory from previous runs')
    shutil.rmtree("temp")
    os.mkdir('temp')
else:
    os.mkdir('temp')

if not os.path.exists('runs'): os.mkdir('runs')

test_data_loader = DataLoader(dataset.test, batch_size=len(dataset.test), shuffle=False, pin_memory = True)
model, optimizer, test_data_loader = accelerator.prepare(model, optimizer, test_data_loader)


if not args.train or os.path.exists(args.model_args):
    if os.path.exists(args.model_args):
        log.info(f"Found model file '{args.model_args}' with trained parameters, restoring model state for inference..")
        model.load_state_dict(torch.load(args.model_args, map_location = device))

        for batch in test_data_loader:
            prediction_bin, logits, stats = predict_homolog_genes(model, None, batch, binary_th=binary_th, dataset = dataset, base_labels = (dataset.base_labels, dataset.base_labels_raw))

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + args.tb_comment
        log.info(f"Moving output to from '{os.path.join('temp', run_id)}/' to '{args.output}/' ...")
        os.mkdir(os.path.join('temp', run_id))
        shutil.copyfile(os.path.join('plots', 'pr_curve.png'), os.path.join('temp', run_id, run_id + 'pr_curve.png'))
        shutil.move(os.path.join('q_score_vs_logit.csv'), os.path.join('temp', run_id, run_id + 'q_score_vs_logit.csv'))
        shutil.move(os.path.join('temp', run_id), args.output)
            
        stats['mode'] = 'test'

    else:
        log.error(f"Could not infer model because model parameters file '{args.model_args}' was not found, exiting.")
        quit()

elif args.train:
    # Training loop

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + args.tb_comment
    writer = SummaryWriter(log_dir = os.path.join('temp', run_id), comment = args.tb_comment)
    
    train_data_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True, pin_memory = True)
    val_data_loader = DataLoader(dataset.val, batch_size=args.batch_size, shuffle=True, pin_memory = True)

    model, optimizer, train_data_loader, scheduler, val_data_loader, test_data_loader = accelerator.prepare(model, optimizer, train_data_loader, scheduler, val_data_loader, test_data_loader)

    log.info(f"Training on device: {device}")

    log.info(f"Entering training loop with batch size: {args.batch_size}, class balance: {dataset.class_balance}, {len(train_data_loader)} batches.")#{dataset.class_balance}.")

    with Progress(transient = True) as progress:

        training_bar = progress.add_task("Epochs completed:", total=args.epochs)
        batch_bar    = progress.add_task("Training model on training data set", total=len(train_data_loader))
        val_bar      = progress.add_task("Infering model on validation data set:", total=len(val_data_loader))

        for epoch in range(args.epochs):

            if 'cuda' in device.type:
                
                torch.cuda.reset_peak_memory_stats()

                if ((epoch + 1) % 10 == 0):
                    log.info('Trying to clear unneccesarily reserved GPU memory..')
                    torch.cuda.empty_cache()

            val_loss, train_loss = 0, 0


            for batch_num, batch in enumerate(train_data_loader):
            #for batch_num, batch in enumerate(dataset.train):
            #for batch_num in range(args.num_batches):

                if oom_count > (args.epochs * len(train_data_loader) * 0.1):
                    log.error(f'Training repeatedly failed because GPU went out of memory. Try with smaller batch size or smaller -n.')
                    quit()
                
                model.train()

                #batch = sub_sample_graph_edges(dataset.train, device, fraction = 0.8) if not args.union_edge_weights else dataset.train
                labels = batch[0].y if isinstance(batch, tuple) else batch.y

                # clear old gradients so only current batch gradients are used
                optimizer.zero_grad()

                # this calls the models forward function since model is callable
                log.debug('Calling forward step on current batch..')

                try:
                    output = model(batch)
                    log.debug('Calling loss function on current batch..')
                    log.debug(batch)
                    loss = criterion(output, labels)
                    log.debug('Calling backward step on current batch..')


                    accelerator.backward(loss)

                except torch.OutOfMemoryError:
                    log.warning(f'OOM error during forward or backward pass on batch {batch_num}, skipping batch...')
                    if 'cuda' in device.type: torch.cuda.empty_cache()
                    oom_count += 1
                    continue

                log.debug('Calling optimizer step on current batch..')
                optimizer.step()
                
                train_loss += loss.item()
                labels = labels.detach()
                probabilities = torch.sigmoid(output.detach())
                binary_prediction_train = (probabilities >= binary_th).int()
                binary_confusion_matrix_train.update(binary_prediction_train, labels)

                del loss
                del output
                del batch
                del labels

                if args.dynamic_binary_threshold:
                    fpr, tpr, thresholds = roc_curve(labels, torch.sigmoid(output.detach()))
                    youden_index = tpr - fpr
                    optimal_threshold = thresholds[youden_index.argmax()]
                    writer.add_scalar("Optimal_Binary_Threshold/val", optimal_threshold, epoch)
                    binary_th = optimal_threshold

                    print(optimal_threshold)
                
                progress.update(batch_bar, advance = 1)


            with torch.no_grad():  # Disable gradient calculation for validation

                for num_batch, batch in enumerate(val_data_loader):

                    model.eval()

                    try:

                        output = model(batch)

                    except torch.OutOfMemoryError:
                        log.warning(f'OOM error during forward pass on evaluation batch {batch_num}, skipping batch...')
                        if 'cuda' in device.type: torch.cuda.empty_cache()
                        oom_count +=1
                        continue

                    val_labels = batch.y

                    loss = criterion(output, val_labels)
                    val_loss += loss.item()
                    
                    probabilities = torch.sigmoid(output.detach())
                    binary_prediction_val = (probabilities >= binary_th).int()
                    
                    val_labels = val_labels.detach().int()
                    binary_confusion_matrix_val.update(binary_prediction_val, val_labels)
                    binary_auroc.update(probabilities, val_labels)
                    binary_average_precision.update(probabilities, val_labels)

                    del output
                    del loss
                    del batch
                    del val_labels
                    
                    progress.update(val_bar, advance = 1)


            # val metrics
            conf_matrix = binary_confusion_matrix_val.compute()
            tn = conf_matrix[0, 0].item()
            fp = conf_matrix[0, 1].item()
            fn = conf_matrix[1, 0].item()
            tp = conf_matrix[1, 1].item()
            binary_confusion_matrix_val.reset()

            roc_auc_val = binary_auroc.compute().item()
            binary_auroc.reset()
            pr_auc_val = binary_average_precision.compute().item()
            binary_average_precision.reset()

            precision_val = tp / (tp + fp + 1e-10)
            recall_val = tp / (tp + fn + 1e-10)
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
            acc_val = (tp + tn) / (tp + tn + fp + fn)

            scheduler.step(val_loss/len(val_data_loader))

            writer.add_scalar("ROC-AUC/val", roc_auc_val, epoch)
            writer.add_scalar("PR-AUC/val", pr_auc_val, epoch)
            writer.add_scalar("Loss/val", val_loss/len(val_data_loader), epoch)
            writer.add_scalar("Acc/val", acc_val, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("precision/val", precision_val, epoch)
            writer.add_scalar("recall/val", recall_val, epoch)
            writer.add_scalar("F1/val", f1_val, epoch)

            #train metrics
            conf_matrix = binary_confusion_matrix_train.compute()
            tn = conf_matrix[0, 0].item()
            fp = conf_matrix[0, 1].item()
            fn = conf_matrix[1, 0].item()
            tp = conf_matrix[1, 1].item()
            binary_confusion_matrix_train.reset()

            precision_train = tp / (tp + fp + 1e-10)
            recall_train = tp / (tp + fn + 1e-10)
            f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-10)
            acc_train = (tp + tn) / (tp + tn + fp + fn)

            writer.add_scalar("Loss/train", train_loss/len(train_data_loader), epoch)
            writer.add_scalar("Acc/train", acc_train, epoch)
            writer.add_scalar("F1/train", f1_train, epoch)
        
            # get some metrics
            log.info(f"Epoch: {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.10f}, Val AP: {pr_auc_val:.4f}")
            log.info(f"Train Loss: {train_loss/len(train_data_loader):.4f}, Train Acc: {acc_train:.4f}, Train F1: {f1_train:.4f}, Val   Loss: {val_loss/len(val_data_loader):.4f}, Val Acc  : {acc_val:.4f}, Val F1  : {f1_val:.4f}")
            if 'cuda' in device.type: log.info(f"GPU max mem allocated: {torch.cuda.max_memory_allocated('cuda:0') / 1024**2} MB, GPU mem reserved: {torch.cuda.memory_reserved('cuda:0') / 1024**2} MB.")
            
            progress.update(training_bar, advance = 1)
            progress.reset(batch_bar)
            progress.reset(val_bar)

    if 'cuda' in device.type and ((epoch + 1) % 10 == 0):
        log.info('Trying to clear unneccesarily reserved GPU memory..')
        torch.cuda.empty_cache()

    #plot_loss_accuracy(args.epochs, train_losses, train_accuracies, val_losses, val_accuracies, f1_train_lst)
    log.info('Unwrapping model from accelerate layers.')
    model = accelerator.unwrap_model(model)
    log.info(f"Saving model to file '{os.path.join('temp', run_id, args.model_args)}'")
    torch.save(model.state_dict(), os.path.join('temp', run_id, args.model_args))

    # get metrics on test dataset
    for batch in test_data_loader:
        prediction_bin, prediction_scores, stats = predict_homolog_genes(model, None, batch, binary_th=binary_th, base_labels = (dataset.base_labels, dataset.base_labels_raw), dataset =  dataset)
        writer.add_pr_curve('PR/test', batch.y.cpu(), torch.sigmoid(prediction_scores))

    stats['simulate_dataset'] = 0
    hparams['simulate_dataset'] = 0
    hparams['class_balance'] = hparams['class_balance'].item()
    writer.add_hparams(hparams, stats)
    log.info(f"Time elapsed: {format_duration(time.time() - start)}.")

    stats['binary_threshold'] = binary_th
    stats['date'] = str(datetime.date.today())
    stats['neighbours'] = args.neighbours
    stats['mode'] = 'train'
    stats['epochs'] = args.epochs
    stats['batch_size'] = args.batch_size
    stats['device'] = device
    stats['runtime'] = (time.time() - start)
    stats['num_genomes'] = num_genomes
    write_stats_csv(stats)
    writer.flush()
    
    log.info(f"Moving output from '{os.path.join('temp', run_id, run_id)}' to '{args.output}' ...")
    shutil.copyfile(os.path.join('plots', 'pr_curve.png'), os.path.join('temp', run_id, run_id + 'pr_curve.png'))
    
    if os.path.exists('q_score_vs_logit.csv'):
        shutil.move(os.path.join('q_score_vs_logit.csv'), os.path.join('temp', run_id, run_id + 'q_score_vs_logit.csv'))
    shutil.move(os.path.join('temp', run_id), args.output)
    #shutil.move(os.path.join('pangnn.log', run_id), 'runs')
    writer.close()

#write_groups_file(dataset.test, prediction_bin)
# map quality Q score transform
# test different architectures / scores on sim data, then introduce hard cases in sim data and 
# test again, 8-20 genomes
# shuffle gene synteny (by block), hardest case - just shuffle all gibs sampling
# compare metrics, auc, pr, f1 etc..
# improvement?

# Stop cProfile for time profiling
#profiler.disable()

# Process and print time stats
#stats = pstats.Stats(profiler)
#stats.strip_dirs()
#stats.sort_stats("tottime").print_stats(20)