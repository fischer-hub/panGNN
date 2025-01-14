import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import os
from sklearn.metrics import roc_curve, auc


def plot_simscore_class(dataset, path = os.path.join('plots', 'score_class.png')):
    """Plot 


    Args:
        labels (tensor): 
    """

    sim_score = dataset.edge_weight_ts
    labels = dataset.labels_ts
    plt.figure()
    plt.scatter(list(labels), list(sim_score), alpha=0.7, c=labels, cmap='coolwarm', edgecolor='k')

    # Add labels and title
    plt.xticks([0, 1], labels=["Heterolog Gene Pair", "Homolog Gene Pair"])
    plt.xlabel("Class")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Score Distribution by Class")

    # Add grid
    plt.grid(alpha=0.3)
    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    plt.savefig(path)

    plt.title("Log Transformed Similarity Score Distribution by Class")
    plt.ylabel("Log Similarity Score")
    plt.yscale('log')  # Apply log scale to y-axis
    plt.savefig(f"{path.replace('.png', '_log.png')}")


    

def plot_roc(labels, probabilities, path = os.path.join('plots', 'roc.png')):
    """Plot operator-receiver-curve plot for 'ground truth' labels and predicted
    probabilities.

    Args:
        labels (tensor): labels describing whether or not two nodes are connected by an edge ('ground truth')
        probabilities (tensor): predicted probability for two nodes to be connected by an edge
        path (path): path to save roc-plot file to
    
    Returns:
        roc_auc (float): fraction of area under the (roc) curve
    """

    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    plt.savefig(path)

    return roc_auc


def plot_loss_accuracy(num_epochs, train_losses, train_accuracies, path = os.path.join('plots', 'loss_acc.png')):
    # Plot Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='g', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    plt.savefig(path)


def plot_graph(dataset, path):
    """Plot a graph as png based on input dataset consisting of node tensor,
    edge index and egde attribute tensor.

    Args:
        dataset (Data object): data object containing node tensor,
    edge index and egde attribute tensor.
    """

    edge_labels = dataset.edge_attr
    plt.figure(3,figsize=(12,12)) 

    # convert dataset to NetworkX graph
    G = to_networkx(dataset, edge_attrs=['edge_attr'], node_attrs=['x'])
    # map integer gene ids back to original string ids
    egde_label_mapping = {i: dataset.gene_ids_lst[i] for i in range(len(dataset.gene_ids_lst))}
    G = nx.relabel_nodes(G, egde_label_mapping)

    # whatever this does
    pos = nx.spring_layout(G)  # Layout for visualization

    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    # Draw edges with weights
    nx.draw_networkx_edges(G, pos, width=2)
    edge_labels = {(u, v): f"{d['edge_attr']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.25)

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path)


def plot_logit_distribution(logits, path = os.path.join('plots', 'logit_distribution.png')):

    plt.figure()

    try:
        values = logits.numpy()
    except Exception:
        values = logits

    plt.hist(values, bins=15, range = (min(values), max(values)))

    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path)