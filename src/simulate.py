import random, torch
from src.setup import log
import numpy as np


def simulate_bit_scores(expectation_value, dispersion, bit_scores_lst, indices = None):

    # Calculate shape (k) and scale (theta) from mean and variance
    shape = expectation_value / dispersion
    scale = expectation_value


    if indices:
        # Sample from the Beta distribution
        scores = np.random.gamma(shape, scale, size = len(indices))

        for idx, score in zip(indices, scores):
            bit_scores_lst[idx] = score

        return bit_scores_lst
    else:
        scores = np.random.gamma(shape, scale, size = len(bit_scores_lst))
    
        return scores


def generate_data(num_genes_per_genome, num_gene_families1, num_gene_families2, fraction_orthologs, fraction_paralogs_same_species, fraction_paralogs_diff_species):

    # generate integer node IDs
    genome_1 = [f"G1_{id}" for id in range(1, num_genes_per_genome)]
    random.shuffle(genome_1)
    genome_2 = [f"G2_{id}" for id in range(1, num_genes_per_genome)]
    random.shuffle(genome_2)


    print(genome_1)

    gene_id_integer_dict1 = {gene: idx for idx, gene in enumerate(genome_1)}
    gene_id_integer_dict2 = {gene: idx for idx, gene in enumerate(genome_2)}
    gene_ids1_lst = list(gene_id_integer_dict1.values())
    gene_ids2_lst = list(gene_id_integer_dict2.values())

    edge_index = torch.stack((torch.tensor(gene_ids1_lst), torch.tensor(gene_ids2_lst)))
    labels_lst = [0] * num_genes_per_genome
    edge_weights_lst = [0] * num_genes_per_genome
    background_noise = simulate_bit_scores(20, 5, edge_weights_lst)

    # reserve nodes for gene familie simulation
    gene_families_1 = random.sample(genome_1, num_gene_families1)
    gene_families_2 = random.sample(genome_2, num_gene_families2)

    # ranomdly sample indices of node pairs in the edge index that will be simulated orthologs
    num_orthologs = int(fraction_orthologs * num_genes_per_genome)
    ortholog_indices = random.sample(range(num_genes_per_genome)[:num_orthologs], num_orthologs)
    edge_weights_lst = simulate_bit_scores(100, 10, background_noise, ortholog_indices)
    
    num_paralogs = int(fraction_paralogs_same_species * num_genes_per_genome)
    paralogs_same_indices = random.sample(range(num_genes_per_genome)[:num_orthologs + num_paralogs], num_paralogs)
    edge_weights_lst = simulate_bit_scores(70, 10, edge_weights_lst, paralogs_same_indices)
    
    num_paralogs_diff = int(fraction_paralogs_diff_species * num_genes_per_genome)
    paralogs_diff_indices = random.sample(range(num_genes_per_genome)[:num_orthologs + num_paralogs + num_paralogs_diff], num_paralogs_diff)
    edge_weights_lst = simulate_bit_scores(80, 10, edge_weights_lst, paralogs_diff_indices)

    tmp_lst = list(zip(labels_lst, edge_weights_lst))
    random.shuffle(tmp_lst)
    labels_lst, edge_weights_lst = zip(*tmp_lst)

    return (gene_ids1_lst + gene_ids2_lst, edge_index, edge_weights_lst, labels_lst)

    import matplotlib.pyplot as plt



    #plt.hist(ortholog_similarity_scores, bins=30, density=True, alpha=0.7, label="Simulated Ortholog Similarity Scores")
    #plt.title("Simulated Ortholog Similarity Scores from Beta Distribution")
    #plt.legend()
    #plt.show()

    quit()

