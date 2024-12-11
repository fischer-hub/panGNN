import random, torch, os
from src.setup import log
import numpy as np
from src.plot import plot_logit_distribution, plot_simscore_class
from src.dataset import Data, HomogenousDataset
from src.preprocessing import construct_neighbour_lst, generate_neighbour_edge_features



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


    gene_id_integer_dict1 = {gene: idx for idx, gene in enumerate(genome_1)}
    gene_id_integer_dict2 = {gene: idx for idx, gene in enumerate(genome_2)}
    gene_ids1_lst = list(gene_id_integer_dict1.values())
    gene_ids2_lst = list(gene_id_integer_dict2.values())

    edge_index = torch.stack((torch.tensor(gene_ids1_lst), torch.tensor(gene_ids2_lst)))
    labels_lst = [0] * num_genes_per_genome
    edge_weights_lst = [0] * num_genes_per_genome
    neighbour_weight_lst = [0] * num_genes_per_genome
    background_noise = simulate_bit_scores(20, 5, edge_weights_lst)

    # reserve nodes for gene familie simulation
    gene_families_1 = random.sample(genome_1, num_gene_families1)
    gene_families_2 = random.sample(genome_2, num_gene_families2)

    # ranomdly sample indices of node pairs in the edge index that will be simulated orthologs
    num_orthologs = int(fraction_orthologs * num_genes_per_genome)
    ortholog_indices = random.sample(range(num_genes_per_genome)[:num_orthologs], num_orthologs)
    edge_weights_lst = simulate_bit_scores(500, 10, background_noise, ortholog_indices)
    neighbour_weight_lst = simulate_bit_scores(500, 10, neighbour_weight_lst, ortholog_indices)
    for i in ortholog_indices: labels_lst[i] = 1

    num_in_paralogs = int(fraction_paralogs_same_species * num_genes_per_genome)
    paralogs_in_indices = random.sample(range(num_genes_per_genome)[:num_orthologs + num_in_paralogs], num_in_paralogs)
    edge_weights_lst = simulate_bit_scores(400, 10, edge_weights_lst, paralogs_in_indices)
    neighbour_weight_lst = simulate_bit_scores(25, 10, neighbour_weight_lst, paralogs_in_indices)
    #for i in paralogs_in_indices: labels_lst[i] = 1
    
    num_out_paralogs = int(fraction_paralogs_diff_species * num_genes_per_genome)
    paralogs_out_indices = random.sample(range(num_genes_per_genome)[:num_orthologs + num_in_paralogs + num_out_paralogs], num_out_paralogs)
    edge_weights_lst = simulate_bit_scores(450, 10, edge_weights_lst, paralogs_out_indices)
    neighbour_weight_lst = simulate_bit_scores(20, 10, neighbour_weight_lst, paralogs_out_indices)
    #for i in paralogs_out_indices: labels_lst[i] = 1


    tmp_lst = list(zip(labels_lst, edge_weights_lst, neighbour_weight_lst))
    random.shuffle(tmp_lst)
    labels_lst, edge_weights_lst, neighbour_weight_lst = zip(*tmp_lst)
    neighbour_weight_lst = list(neighbour_weight_lst)
    labels_lst = list(labels_lst)
    edge_weights_lst = list(edge_weights_lst)

    plot_logit_distribution(edge_weights_lst, os.path.join('plots', 'simulated_sim_scores.png'))
    dataset = HomogenousDataset()
    dataset.x = gene_ids1_lst + gene_ids2_lst
    dataset.edge_weight_ts = torch.tensor(edge_weights_lst)
    dataset.labels_ts = torch.tensor(labels_lst)
    dataset.y = torch.tensor(labels_lst)
    dataset.neighbour_edge_weights_ts =torch.tesnor(neighbour_weight_lst)
    plot_simscore_class(dataset, os.path.join('plots', 'simulated_sim_score_class.png'))


    return dataset