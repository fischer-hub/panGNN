import torch
from rich.console import Console
from src.setup import log
from src.plot import plot_roc
from sklearn.metrics import confusion_matrix

def predict_homolog_genes(model, dataset, binary_th = 0.5):
    """Infer the GNN with given trained model and predict homolog genes from
    input similarity graph.

    Args:
        model (class 'GCN'): model with learned parameters to infer on the input graph
        dataset (class 'Data'): dataset containing the input graph, with structures
                                node tensor, edge index, edge attribute tensor and
                                y (labels tensor, optionally used for training)

    Returns:
        pred (tensor): tensor containing the final predictions for each pair of 
                       nodes represented in the edge index
        
    """
    model.eval()
    with torch.no_grad():
        with Console().status("Infering model on test data..") as status:
            edge_scores = model(dataset)
        
        if hasattr(dataset, 'y'):
            log.info('Calculating metrics..')
            probablilities = torch.sigmoid(edge_scores)
            binary_prediction = torch.tensor((probablilities >= binary_th).int())
            print(probablilities)
            print(binary_prediction)
            auc = plot_roc(dataset.y, probablilities)
            conf_matrix = confusion_matrix(dataset.y, binary_prediction)
            tn, fp, fn, tp = conf_matrix.ravel()

            accuracy = ((binary_prediction == dataset.y).sum().item()) / len(dataset.y)
            log.info("\n\n----------METRICS----------\n")
            log.info(f"Correctly predicted: {(binary_prediction == dataset.y).sum().item() } out of {len(dataset.y)} genes to be homologs.")
            log.info(f"AUC on test dataset: {auc}")
            log.info(f"Accuracy on test dataset: {accuracy}")
            log.info(f"Accuracy2 on test dataset: {(tp+tn)/(tp+fp+tn+fn)}")
            log.info(f"Precision on test dataset: {tp/(tp+fp)}")
            log.info(f"Recall (sens) on test dataset: {tp/(tp+fn)}")
            log.info(f"Specifity on test dataset: {tn/(fp+tn)}")
            log.info(f"F1 on test dataset: {2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))}")
            log.info(f"Got confusion matrix:\n\n\n{'':15}|{'pred negative':^15}|{'pred positive':^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label negative':15}|{conf_matrix[0][0]:^15}|{conf_matrix[0][1]:^15}")
            log.info(f"---------------------------------------------")
            log.info(f"{'label positive':15}|{conf_matrix[1][0]:^15}|{conf_matrix[1][1]:^15}\n\n")
        else:
            log.error('No labels supplied, can not calculate metrics.')
        log.info('Exiting.')
    
    return (binary_prediction, edge_scores)
