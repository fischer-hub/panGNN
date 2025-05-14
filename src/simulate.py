import random, torch, os
from src.setup import log
import numpy as np
from src.plot import plot_logit_distribution, plot_simscore_class
from src.preprocessing import construct_neighbour_lst, generate_neighbour_edge_features
from src.helper import char_id_generator, pairwise
from collections import defaultdict



def simulate_bit_scores(expectation_value, dispersion, n):

    # Calculate shape (k) and scale (theta) from mean and variance
    shape = expectation_value / dispersion
    scale = expectation_value
    return np.random.gamma(shape, scale, size = n)



"""
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
    dataset.neighbour_edge_weights_ts =torch.tensor(neighbour_weight_lst)
    plot_simscore_class(dataset, os.path.join('plots', 'simulated_sim_score_class.png'))


    return dataset
"""

def simulate_gene_ids(num_genes, num_genomes):

    genome_ids = [ None ] * num_genomes
    num_genes_per_genome = int(num_genes / num_genomes)
    
    for idx, genome_id in enumerate(char_id_generator()):
        
        if idx == num_genomes:
            break

        genome_ids[idx] = genome_id

    
    genome_ids_flat = [ f'{genome_id}_{gene_number:06}' for genome_id in genome_ids for gene_number in range(num_genes_per_genome) ]
    genome_ids_by_genome = [[ f'{genome_id}_{gene_number:06}' for gene_number in range(num_genes_per_genome)] for genome_id in genome_ids]
    return genome_ids_flat, genome_ids_by_genome


def simulate_similarity_scores_and_ribap_dict(gene_lsts, frac_pos_edges):

    similarity_dict = defaultdict(dict)
    ribap_groups_dict = {}
    edge_count = 0
    
    num_genomes = len(gene_lsts)
    num_genes_per_genome = len(gene_lsts[0])
    num_pos_edges = (num_genomes-1) * num_genes_per_genome
    num_total_edges = num_pos_edges / frac_pos_edges
    num_negative_edges = num_total_edges - num_pos_edges
    mean_num_negative_edges_per_gene = num_negative_edges / num_pos_edges
    num_negative_edges_per_gene_lst = np.random.normal(mean_num_negative_edges_per_gene, 5, num_pos_edges)
    num_negative_edges_per_gene_lst = [int(i) for i in num_negative_edges_per_gene_lst]
    ribap_groups_lst = [none] * num_genes_per_genome
    ribap_group_count = 0
    

    for group in zip(*gene_lsts):

        # genome idx of target genes
        target_genome_idx = 1
        
        ribap_groups_lst[ribap_group_count] = group
        ribap_group_count += 1

        for key_gene in group:
            ribap_groups_dict[key_gene] = [gene for gene in group if isinstance(gene, str) and key_gene not in gene]


        # assume genes at same pos are orthologs and score come from gamma distr with highest mean
        ortholog_scores = simulate_bit_scores(500, 10, len(group))

        for (source, target), score in zip(pairwise(group), ortholog_scores):

            # raw sim score is not direction independent but we assume this for simplicity
            similarity_dict[source] = {target: score}
            similarity_dict[target] = {source: score}

            # for source gene select n target genes to add negative edges to
            negative_edge_idxs = random.choices(range(num_genes_per_genome), k = num_negative_edges_per_gene_lst[edge_count])
            heterolog_scores = simulate_bit_scores(20, 10, len(negative_edge_idxs))

            for negative_edge_idx, score in zip(negative_edge_idxs, heterolog_scores):

                # add heterolog sim scores to sim score dict both ways
                target = gene_lsts[target_genome_idx][negative_edge_idx]
                similarity_dict[source] = {target: score}
                similarity_dict[target] = {source: score}

            edge_count += 1
            target_genome_idx += 1

    assert len(similarity_dict) == (num_total_edges * 2), f'Number of similarity scores in dictionary ({len(similarity_dict)}) is not equal to number of expected similarity edges ({num_total_edges * 2}).'
    assert len(ribap_groups_dict) == (num_genes_per_genome * (num_genomes - 1)) , f'Number of ribap group mappings found ({len(ribap_groups_dict)}) is not equal to number of expected ribap group mappings ({(num_genes_per_genome * (num_genomes - 1)}).'
    assert len(ribap_groups_lst) == num_genes_per_genome , f'Number of ribap groups ({len(ribap_groups_lst)}) is not equal to number of genes per genome ({num_genes_per_genome}), but every gene should belong to one RIBAP group.'

    return similarity_dict, ribap_groups_dict, ribap_groups_lst