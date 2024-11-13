import torch
from rich.console import Console
from src.setup import log

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
            binary_prediction = torch.tensor((torch.sigmoid(edge_scores) >= binary_th).int())
            accuracy = ((binary_prediction == dataset.y).sum().item()) / len(dataset.y)
            log.info(f"Correctly predicted: {(binary_prediction == dataset.y).sum().item() } out of {len(dataset.y)} genes to be homologs.")
            log.info(f"Accuracy on test dataset: {accuracy}")
        else:
            log.error('No labels supplied, can not calculate accuracy.')
        log.info('Exiting.')
    
    return (binary_prediction, edge_scores)
