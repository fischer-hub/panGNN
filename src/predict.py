import torch, time
from rich.console import Console
from src.setup import log, args
from src.plot import plot_roc, plot_logit_distribution, plot_pr_curve, plot_confusion_matrix, plot_sim_score_vs_logit
from src.helper import concat_graph_data, calculate_logit_baseline_labels, format_duration
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay
import torch.nn.functional as F


def predict_homolog_genes(model, train_dataset = None, test_dataset = None, binary_th = 0.72, base_labels = None, refined_base_labels = None, dataset = None):
    """Infer the GNN with given trained model and predict homolog genes from
    input similarity graph.

    Args:
        model (class 'GCN'): model with learned parameters to infer on the input graph
        test_dataset (class 'Data'): test_dataset containing the input graph, with structures
                                node tensor, edge index, edge attribute tensor and
                                y (labels tensor, optionally used for training)

    Returns:
        pred (tensor): tensor containing the final predictions for each pair of 
                       nodes represented in the edge index
        
    """
    stats = {}
    #model.to(device)
    model.eval()
    with torch.no_grad():
        with Console().status("Infering model on test data..") as status:
            #test_dataset.to(device)
            inference_start_time = time.time()
            edge_scores = model(test_dataset)
            log.info(f"Time elapsed during inference on test datset: {format_duration(time.time() - inference_start_time)}.")

            if isinstance(test_dataset, tuple): test_dataset = test_dataset[0]

            if train_dataset:
                #train_dataset.to(device)
                train_dataset = concat_graph_data(train_dataset)
                edge_scores_train = model(train_dataset)
                if isinstance(train_dataset, tuple): train_dataset = train_dataset[0]
        
        if hasattr(test_dataset, 'y'):
            log.info('Calculating metrics..')
            
            if train_dataset:
                probablilities_train = torch.sigmoid(edge_scores_train)
                binary_prediction_train = (probablilities_train >= binary_th).int()


            probablilities = torch.sigmoid(edge_scores).cpu()
            binary_prediction = (probablilities >= binary_th).int()

            test_labels = test_dataset.y.cpu()

            auc, opt_th = plot_roc(test_labels, probablilities)
            stats['auc_test'] = auc
            stats['optimatl_threshold'] = opt_th

            conf_matrix = confusion_matrix(test_labels, binary_prediction, labels = [0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()

            stats['tn'] = tn
            stats['fp'] = fp
            stats['fn'] = fn
            stats['tp'] = tp

            plot_logit_distribution(edge_scores.cpu(), 'plots/logit_dist.png')
            plot_logit_distribution(probablilities, 'plots/prob_hist.png')

            random_pred = torch.randint(0,2,(len(binary_prediction),))

            if not args.train:
                log.info('Calculating max logit candidate baseline..')
                max_candidate_logit_labels = calculate_logit_baseline_labels(test_dataset, dataset.sim_score_dict, edge_scores, dataset.gene_str_ids_lst, dataset.gene_id_position_dict)
                AP = plot_pr_curve(test_labels, probablilities, base_labels, refined_base_labels, max_candidate_logit_labels = max_candidate_logit_labels)
                plot_confusion_matrix(test_labels, base_labels[0], title='Q-Score Max Candidate', path = 'plots/q_score_conf_matrix.png')
                plot_confusion_matrix(test_labels, base_labels[1], title='Raw Score Max Candidate', path = 'plots/raw_score_conf_matrix.png')
                plot_confusion_matrix(test_labels, max_candidate_logit_labels, title='Max Logit Candidate', path = 'plots/logit_conf_matrix.png')
                plot_sim_score_vs_logit(test_labels, test_dataset.edge_attr, edge_scores, test_dataset.edge_index, dataset.gene_str_ids_lst, base_labels, max_candidate_logit_labels)
            else:
                AP = plot_pr_curve(test_labels, probablilities, base_labels, refined_base_labels)

            stats['average_precision'] = AP

            accuracy_test = ((binary_prediction == test_labels).sum().item()) / len(test_labels)
            stats['acc_test'] = accuracy_test
            stats['acc_train'] = 0

            if train_dataset: 
                accuracy_train = ((binary_prediction_train == train_dataset.y).sum().item()) / len(train_dataset.y)
                stats['acc_train'] = accuracy_train

            guess = ((random_pred == test_labels).sum().item()) / len(test_labels)
            log.info("\n\n----------METRICS----------\n")
            log.info(f"Correctly predicted: {(binary_prediction == test_labels).sum().item() } out of {len(test_labels)} edges.")
            log.info(f"AUC on test dataset: {auc}")
            log.info(f"Average precision on test dataset: {AP}")
            log.info(f"Accuracy on test dataset: {accuracy_test}")
            log.info(f"Accuracy on test data from conf mat: {(tp + tn) / (tp + tn + fp + fn)}")
            if train_dataset:
                log.info(f"Accuracy on train dataset: {accuracy_train}")
            log.info(f"Accuracy when guessing: {guess}")
            log.info(f"Precision on test dataset: {tp/(tp+fp)}")
            stats['precision'] = tp/(tp+fp)
            log.info(f"Recall (sens) on test dataset: {tp/(tp+fn)}")
            stats['recall'] = tp/(tp+fn)
            log.info(f"Specifity on test dataset: {tn/(fp+tn)}")
            stats['specifity'] = tn/(fp+tn)
            log.info(f"F1 on test dataset: {2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))}")
            stats['f1'] = 2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))
            log.info(f"Got confusion matrix:\n\n\n{'':15}|{'pred negative':^15}|{'pred positive':^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label negative':15}|{round(conf_matrix[0][0]/len(test_labels)*100, 2):^15}|{round(conf_matrix[0][1]/len(test_labels)*100, 2):^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label positive':15}|{round(conf_matrix[1][0]/len(test_labels)*100, 2):^15}|{round(conf_matrix[1][1]/len(test_labels)*100, 2):^15}\n\n")
        else:
            log.error('No labels supplied, can not calculate metrics.')
    
    return (binary_prediction, edge_scores, stats)
