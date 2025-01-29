import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import os, umap
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay
import numpy as np
from sklearn.decomposition import PCA
import torch

def plot_umap_pca(dataset, path = os.path.join('plots', 'umap.png')):

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
    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    plt.savefig(path)


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
    youden_index = tpr - fpr
    optimal_threshold = thresholds[youden_index.argmax()]

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f}), opt. th {optimal_threshold:.2f}')
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



def plot_pr_curve(labels, probabilities, path = os.path.join('plots', 'pr_curve.png')):

    plt.figure(figsize=(12, 5))
    display = PrecisionRecallDisplay.from_predictions(labels, probabilities, name="PR", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    plt.savefig(path)



def plot_loss_accuracy(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, f1_train_lst, path = os.path.join('plots', 'loss_acc.png')):
    
    f1_train_lst = np.array(f1_train_lst)
    f1_train_lst[np.isnan(f1_train_lst)] = 0
    
    # Plot Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', color='y', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='g', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', color='y', label='Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), f1_train_lst, marker='o', color='r', label='Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy and F1 Score over Epochs')
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


def plot_simscore_distribution_by_class(dataset, path = os.path.join('plots', 'sim_score_distribution_by_class.png')):

    plt.figure()
    import statistics
    class_0_sim_scores = [dataset.edge_attr[i].item() for i in range(len(dataset.edge_attr)) if dataset.y[i] == 0]
    print(f"class 0 sim scores: stdev: {statistics.stdev(class_0_sim_scores)}, mean: {sum(class_0_sim_scores) / len(class_0_sim_scores)}")
    

    class_1_sim_scores = [dataset.edge_attr[i].item() for i in range(len(dataset.edge_attr)) if dataset.y[i] == 1]
    print(f"class 1 sim scores: stdev: {statistics.stdev(class_1_sim_scores)}, mean: {sum(class_1_sim_scores) / len(class_1_sim_scores)}")

    plt.hist(class_0_sim_scores, bins=15, label = 'class 0', alpha = 0.6)
    plt.hist(class_1_sim_scores, bins=15, label = 'class 1', alpha = 0.6)
    plt.legend(loc='upper right')

    plt.xlabel('Score Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores by Class')

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path)



def plot_union_graph(dataset, path):
    """Plot a graph as png based on input dataset consisting of node tensor,
    edge index and egde attribute tensor.

    Args:
        dataset (Data object): data object containing node tensor,
    edge index and egde attribute tensor.
    """

    dataset.edge_index = dataset.union_edge_index
    dataset.edge_attr = torch.cat((dataset.edge_attr, torch.tensor([1] * (len(dataset.edge_index[0]) - len(dataset.edge_attr)))))
    edge_labels = dataset.edge_attr
    plt.figure(3,figsize=(12,12)) 


    # convert dataset to NetworkX graph
    G = to_networkx(dataset, edge_attrs=['edge_attr'], node_attrs=['x'])
    # map integer gene ids back to original string ids
    node_label_mapping = {i: i+1 for i in range(len(dataset.x))}
    G = nx.relabel_nodes(G, node_label_mapping)

    # whatever this does
    #pos = nx.spring_layout(G)  # Layout for visualization


    # Generate grid positions using grid_2d_graph logic
    pos = {1 : (0, 2), 2 : (1, 2), 3 : (2, 2), 4 : (3, 2),
           5: (0, 1), 6: (1, 1), 7: (2, 1), 8: (3, 1),
           9: (0, 0), 10: (1, 0), 11: (2, 0), 12: (3, 0)}

    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    # Draw edges with weights
    nx.draw_networkx_edges(G, pos, width=2)
    edge_labels = {(u, v): f"{d['edge_attr']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path)