import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx
import os
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import torch, statistics
from src.setup import log

"""
def plot_umap_pca(dataset, path = os.path.join('plots', 'umap.png')):

    umap_model = umap.UMAP(n_components=3, random_state=42)
    dist = []
    for origin_node, target_node in zip(dataset.test.edge_index[0], dataset.test.edge_index[1]):
        dist.append(abs(dataset.test.x[origin_node] - dataset.test.x[target_node]))

    plt.figure()
    plt.scatter(list(dataset.test.edge_attr), dist, alpha=0.7, c=dataset.test.y, cmap='coolwarm', edgecolor='k')

    # Add labels and title
    plt.xlabel("similarity score")
    plt.ylabel("absolute difference in gene position")
    plt.title("similarty score by gene position difference")
    plt.legend()

    plt.show()

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
"""


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

    return (roc_auc, optimal_threshold)



def plot_pr_curve(labels, probabilities, base_labels, refined_base_labels, max_candidate_logit_labels = None, path = os.path.join('plots', 'pr_curve.png'), min_max_raw = None, min_max_q = None):

    labels = labels.tolist()
    AP = average_precision_score(labels, probabilities)
    plt.figure(figsize=(12, 5))
    display = PrecisionRecallDisplay.from_predictions(labels, probabilities, name="PR-logit", plot_chance_level=True, pos_label = 1)
    ax = display.ax_
    _ = display.ax_.set_title("Binary Precision-Recall Curve")
    display.line_.set_color("blue")

    if min_max_raw is not None:
        display_raw_score = PrecisionRecallDisplay.from_predictions(labels, min_max_raw, ax = ax, name="PR-raw-score", plot_chance_level=False, pos_label = 1)
        display_raw_score.line_.set_linestyle('--')
        display_raw_score.line_.set_color('orange')

    if min_max_q is not None:
        display_q_score = PrecisionRecallDisplay.from_predictions(labels, min_max_q, ax = ax, name="PR-Q-score", plot_chance_level=False, pos_label = 1)
        display_q_score.line_.set_linestyle('--')
        display_q_score.line_.set_color('green')

    if max_candidate_logit_labels is not None:
        #logit_precision, logit_recall, _ = precision_recall_curve(labels, max_candidate_logit_labels)
        logit_precision = precision_score(labels, max_candidate_logit_labels, pos_label = 1)
        log.info(f'F1 score of max candidate logit baseline: {f1_score(labels, max_candidate_logit_labels)}')

        logit_recall = recall_score(labels, max_candidate_logit_labels, pos_label = 1)
        #plt.plot(logit_recall, logit_precision, linestyle='--', label='Max Logit Candidate', color='purple')
        #plt.hlines(logit_precision, 0, 1, linestyle='--', label='Max Logit Candidate', color='purple')
        plt.plot(logit_recall, logit_precision, 'D', color = 'blue',  label='Max Logit Candidate', alpha = 0.5)

    if base_labels is not None:
        base_labels, base_labels_raw = base_labels
        baseline_precision, baseline_recall, th = precision_recall_curve(labels, base_labels)
        baseline_precision = precision_score(labels, base_labels, pos_label = 1)
        baseline_recall = recall_score(labels, base_labels, pos_label = 1)
        log.info(f'F1 score of max candidate Q-score transformed baseline:  {f1_score(labels, base_labels)}')
        plt.plot(baseline_recall, baseline_precision, 'D', color = 'green',  label='Max Q-Score Candidate', alpha = 0.5)

        baseline_precision_raw, baseline_recall_raw, _ = precision_recall_curve(labels, base_labels_raw)
        log.info(f'F1 score of max candidate raw score baseline: {f1_score(labels, base_labels_raw)}')
        baseline_precision_raw = precision_score(labels, base_labels_raw, pos_label = 1)
        baseline_recall_raw = recall_score(labels, base_labels_raw, pos_label = 1)
        plt.plot(baseline_recall_raw, baseline_precision_raw, 'D', color = 'orange',  label='Max raw Candidate', alpha = 0.5)

        #plt.plot(baseline_recall, baseline_precision, linestyle='--', label='Max Q-Score Candidate', color='red')
        #plt.hlines(baseline_precision, 0, 1, linestyle='--', label='Max Q-Score Candidate', color='red')
        #plt.plot(baseline_recall_raw, baseline_precision_raw, linestyle='--', label='Max Raw Score Candidate', color='green')
        #plt.hlines(baseline_precision_raw, 0, 1, linestyle='--', label='Max Raw Score Candidate', color='green')
        if refined_base_labels is not None: refined_baseline_precision, refined_baseline_recall, _ = precision_recall_curve(labels, refined_base_labels)
        if refined_base_labels is not None: plt.plot(refined_baseline_recall, refined_baseline_precision, linestyle='--', label='refined RBH', color='yellow')
        plt.legend()

    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    log.info(f"Saving PR-curve plot with baselines to '{path}'.")
    plt.savefig(path)

    return AP



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

    plt.hist(values, bins=35, range = (min(values), max(values)))

    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path)


def plot_simscore_distribution_by_class(dataset, path = os.path.join('plots', 'sim_score_distribution_by_class.png')):

    plt.figure()
    import statistics
    print(dataset)
    class_0_sim_scores = [dataset.edge_attr[i].item() for i in range(len(dataset.y)) if dataset.y[i] == 0]
    print(f"class 0 sim scores: stdev: {statistics.stdev(class_0_sim_scores)}, mean: {sum(class_0_sim_scores) / len(class_0_sim_scores)}")
    

    class_1_sim_scores = [dataset.edge_attr[i].item() for i in range(len(dataset.y)) if dataset.y[i] == 1]
    print(f"class 1 sim scores: stdev: {statistics.stdev(class_1_sim_scores)}, mean: {sum(class_1_sim_scores) / len(class_1_sim_scores)}")
    print(f'percentage pos edges: {len(class_1_sim_scores) / len(dataset.y)}')

    plt.hist(class_0_sim_scores, bins=15, label = 'class 0', alpha = 0.6)
    plt.hist(class_1_sim_scores, bins=15, label = 'class 1', alpha = 0.6)
    plt.legend(loc='upper right')

    plt.xlabel('Score Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
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


def plot_violin_distributions(violin_dict_lst, ribap_dict, prob, path = os.path.join('plots', 'normalized_score_violin.png')):

    print('plotting violins')
    data = pd.DataFrame(columns=['temp', 'score', 'homolog'])
    data = data.astype(dtype={'temp': 'str', 'score': 'float64', 'homolog': 'int64'})
    index = 0

    for idx, tup in enumerate(violin_dict_lst):
        print(f'Preparing data for violin plots for dict {idx} of {len(violin_dict_lst)}')
        temp, current_dict = tup
        
        for origin_gene, candidates in current_dict.items():
            for target_gene, score in candidates.items():
                
                if origin_gene not in ribap_dict.keys():
                    continue
                
                label = '1' if target_gene in ribap_dict[origin_gene] else '0'
                data.loc[index] =  [temp, score, label]
                assert score >= 0
                index += 1

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=data, x="temp", y="score", hue="homolog", split=True, inner="quart", cut = 0)
    plt.xlabel('Softmax Normalization Temperature Value')
    plt.ylabel(f"Normalized Similarity {'Probability' if prob else 'Q-Score'}")
    plt.title('Normalized Similarity Score Distributrions over Changing Temperature')

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path, dpi = 300)

 
def plot_homolog_positions(ribap_dict, gene_str_int_dict, path = os.path.join('plots', 'homolog_positions.png')):
    
    average_dist_lst, x, y = [], [], []
    for origin_gene, candidate_lst in ribap_dict.items():

        distance_lst = [abs(gene_str_int_dict[origin_gene] - gene_str_int_dict[candidate]) for candidate in candidate_lst]
        average_dist_lst.append(sum(distance_lst) / len(distance_lst))

        #x += [gene_str_int_dict[origin_gene]] * len(candidate_lst)
        #y += candidate_lst

    """     
    plt.scatter(x, y)
    plt.plot(x, x, color='red', linestyle='-')
    plt.xlabel('Genomic Position of Reference Gene')
    plt.ylabel('Genomic Position of Target Gene')
    plt.title('Change of Homolog Positions in different Genomes') 
    """


    plt.figure(figsize=(8, 6))
    plt.hist(average_dist_lst, bins=35, range = (min(average_dist_lst), max(average_dist_lst)))

    plt.xlabel('Average Distance of Homolog Gene Positions')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Distances Between Homolog Gene Positions')
    
    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path, dpi = 300)


def plot_confusion_matrix(true_labels, predicted_labels, labels=[0, 1], normalize=None, title="Confusion Matrix", path =  os.path.join('plots', 'conf_matrix.png')):
    """
    Plot a confusion matrix using true and predicted labels.
    
    Args:
        true_labels (list or array): Ground truth labels.
        predicted_labels (list or array): Model predicted labels.
        labels (list): Class labels to display in the matrix.
        normalize (str or None): {'true', 'pred', 'all'}, optional normalization.
        title (str): Title of the plot.
    """
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()

    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path, dpi = 300)


def plot_sim_score_vs_logit(labels, edge_weights, logits, edge_index, gene_lst, base_labels, max_logit_labels, path =  os.path.join('plots', 'sim_score_vs_logit.png')):

    base_labels, base_labels_raw = base_labels
    plt.figure(figsize=(8, 6))
    edge_weights = edge_weights.tolist()[:len(logits)]
    logits = logits.tolist()
    labels = labels.tolist()
    one_count = edge_weights.count(1.0) / len(edge_weights)
    scatter = plt.scatter(edge_weights, logits, c = labels)

    plt.xlabel('Input Similarity Scores')
    plt.ylabel('Output Logits')
    plt.title('Input Similarity Scores vs. Output Logits')
    plt.legend(*scatter.legend_elements())
    
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path, dpi = 300)

    data = pd.DataFrame({'score': edge_weights, 'logit': logits, 'homolog': labels})
    data['score_binned'] = pd.cut(data.score, 8)
    data = data.astype(dtype={'score': 'float64', 'logit': 'float64', 'homolog': 'int64'})


    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=data, x="score_binned", y="logit", hue="homolog", split=True, inner="quart", cut = 0)
    plt.xlabel('Input Similarity Score Interval')
    plt.ylabel("Output Logit")
    plt.title('Input Similarity Scores vs. Output Logits')

    path = os.path.splitext(path)[0] + '_violin.png'

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
    plt.savefig(path, dpi = 400)

    data['source_int_id'] = edge_index[0].tolist()
    data['target_int_id'] = edge_index[1].tolist()
    data['q_score_baseline'] = base_labels
    data['raw_baseline'] = base_labels_raw
    data['logit_baseline'] = max_logit_labels
    data['source_str_id'] = [ gene_lst[i] for i in edge_index[0].tolist() ]
    data['target_str_id'] = [ gene_lst[i] for i in edge_index[1].tolist() ]
    data = data[['source_int_id', 'target_int_id', 'source_str_id', 'target_str_id', 'score', 'logit', 'homolog', 'q_score_baseline', 'raw_baseline', 'logit_baseline']]

    log.info(f"Percentage of edge weights with value 1.0: {one_count}")
    log.info(f"Writing 'q_score_vs_logit.csv' ...")
    
    data.to_csv('q_score_vs_logit.csv', index = False)