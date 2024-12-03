import torch
from rich.console import Console
from src.setup import log
from src.plot import plot_roc, plot_logit_distribution
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def predict_homolog_genes(model, test_dataset, binary_th = 0.5, train_dataset = None):
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
    model.eval()
    with torch.no_grad():
        with Console().status("Infering model on test data..") as status:
            edge_scores = model(test_dataset)
            if train_dataset:
                edge_scores_train = model(train_dataset)
        
        if hasattr(test_dataset, 'y'):
            log.info('Calculating metrics..')
            
            if train_dataset:
                probablilities_train = torch.sigmoid(edge_scores_train)
                binary_prediction_train = (probablilities_train >= binary_th).int()

            probablilities = torch.sigmoid(edge_scores)
            binary_prediction = (probablilities >= binary_th).int()
            print(probablilities)
            print(binary_prediction)
            auc = plot_roc(test_dataset.y, probablilities)
            conf_matrix = confusion_matrix(test_dataset.y, binary_prediction)
            tn, fp, fn, tp = conf_matrix.ravel()

            plot_logit_distribution(edge_scores, 'plots/logit_dist.png')
            plot_logit_distribution(probablilities, 'plots/prob_hist.png')


            random_pred = torch.randint(0,2,(len(binary_prediction),))

            accuracy_test = ((binary_prediction == test_dataset.y).sum().item()) / len(test_dataset.y)
            if train_dataset: 
                accuracy_train = ((binary_prediction_train == train_dataset.y).sum().item()) / len(train_dataset.y)
            guess = ((random_pred == test_dataset.y).sum().item()) / len(test_dataset.y)
            log.info("\n\n----------METRICS----------\n")
            log.info(f"Correctly predicted: {(binary_prediction == test_dataset.y).sum().item() } out of {len(test_dataset.y)} edges.")
            log.info(f"AUC on test dataset: {auc}")
            log.info(f"Accuracy on test dataset: {accuracy_test}")
            if train_dataset:
                log.info(f"Accuracy on train dataset: {accuracy_train}")
            log.info(f"Accuracy when guessing: {guess}")
            log.info(f"Precision on test dataset: {tp/(tp+fp)}")
            log.info(f"Recall (sens) on test dataset: {tp/(tp+fn)}")
            log.info(f"Specifity on test dataset: {tn/(fp+tn)}")
            log.info(f"F1 on test dataset: {2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))}")
            log.info(f"Got confusion matrix:\n\n\n{'':15}|{'pred negative':^15}|{'pred positive':^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label negative':15}|{round(conf_matrix[0][0]/len(test_dataset.y)*100, 2):^15}|{round(conf_matrix[0][1]/len(test_dataset.y)*100, 2):^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label positive':15}|{round(conf_matrix[1][0]/len(test_dataset.y)*100, 2):^15}|{round(conf_matrix[1][1]/len(test_dataset.y)*100, 2):^15}\n\n")
        else:
            log.error('No labels supplied, can not calculate metrics.')
        log.info('Exiting.')
    
    return (binary_prediction, edge_scores)
