import random, torch, os, math, itertools
from src.setup import log
import numpy as np
from src.plot import plot_logit_distribution, plot_simscore_class
from src.preprocessing import construct_neighbour_lst, generate_neighbour_edge_features
from src.helper import char_id_generator, pairwise, chunks, nested_len
from collections import defaultdict



def simulate_bit_scores(expectation_value, dispersion, n):

    # Calculate shape (k) and scale (theta) from mean and variance
    #shape = expectation_value / dispersion
    #scale = expectation_value
        
    shape = (expectation_value ** 2) / dispersion
    scale = dispersion / expectation_value
    
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

def simulate_gene_ids(num_genes_per_genome, num_genomes):

    log.info('Generating gene IDs..')

    genome_ids = [ None ] * num_genomes
    
    for idx, genome_id in enumerate(char_id_generator()):
        
        if idx == num_genomes:
            break

        genome_ids[idx] = genome_id

    
    genome_ids_flat = [ f'{genome_id}_{gene_number:06}' for genome_id in genome_ids for gene_number in range(num_genes_per_genome) ]
    genome_ids_by_genome = [[ f'{genome_id}_{gene_number:06}' for gene_number in range(num_genes_per_genome)] for genome_id in genome_ids]
    return genome_ids_flat, genome_ids_by_genome



def simulate_similarity_scores_and_ribap_dict(gene_lsts, frac_pos_edges):

    log.info('Generating simulated similarity scores and RIBAP groups..')

    similarity_dict = defaultdict(dict)
    ribap_groups_dict = {}
    pos_edge_count = 0
    neg_edge_count = 0
    edge_count = 0
    
    num_genomes = len(gene_lsts)
    log.info(f'Num genomes: {num_genomes}')
    num_genes_per_genome = len(gene_lsts[0])
    log.info(f'Num genes per genome: {num_genes_per_genome}')
    num_total_genes = num_genes_per_genome * num_genomes
    log.info(f'Num total genes: {num_total_genes}')
    num_edges_per_group = math.floor(((num_genomes)*(num_genomes-1))/2)
    log.info(f'Num edges per RIBAP group: {num_edges_per_group}')
    num_pos_edges = num_edges_per_group * num_genes_per_genome
    log.info(f'Num pos edges: {num_pos_edges}')
    num_total_edges = math.floor(num_pos_edges / frac_pos_edges)
    log.info(f'Num total edges: {num_total_edges}')
    num_negative_edges = num_total_edges - num_pos_edges
    log.info(f'Num neg edges: {num_negative_edges}')
    mean_neg_candidates_per_gene = math.floor(num_negative_edges / num_total_genes / num_genomes)
    log.info(f'Num avg neg edges per gene: {mean_neg_candidates_per_gene}')
    #num_negative_edges_per_gene_lst = np.random.poisson(lam = mean_neg_candidates_per_gene, size = num_total_genes)
    num_negative_edges_per_gene_lst = np.random.negative_binomial(n=0.2, p=0.2 / (mean_neg_candidates_per_gene + 0.2), size=num_total_genes)
    num_negative_edges_per_gene_lst = [int(i+1) for i in num_negative_edges_per_gene_lst]
    log.info(f'Number of drawn neg edges: {sum(num_negative_edges_per_gene_lst)}')
    ribap_groups_lst = [None] * num_genes_per_genome
    ribap_group_count = 0
    last_source = ''
   

    for group in zip(*gene_lsts):

        # genome idx of target genes, like this the last and first genome will not have any edges between them but its fine (im too lazy to fix this)
        target_genome_idx = 1
        
        ribap_groups_lst[ribap_group_count] = group
        ribap_group_count += 1

        for key_gene in group:
            ribap_groups_dict[key_gene] = [gene for gene in group if isinstance(gene, str) and key_gene != gene]


        # assume genes at same pos are orthologs and score come from gamma distr with highest mean
        ortholog_scores = simulate_bit_scores(500, 10000, num_edges_per_group)


        for (source, target), score in zip(itertools.combinations(group, 2), ortholog_scores):

            if source == target: continue
            if target_genome_idx == len(gene_lsts)+1: target_genome_idx = 0

            # raw sim score is not direction independent but we assume this for simplicity
            similarity_dict[source][target] = score
            similarity_dict[target][source] = score
            pos_edge_count += 2

            if last_source != source:

                # get genome index in genome list from current target genes origin genome
                target_genome_idx = next((i for i, x in enumerate(group) if x.startswith(target.split('_')[0])), None)

                if target_genome_idx is None: log.error(f'Could not infer origin genome of target gene {target} from its name. Too bad.')
                last_source = source

                # for source gene select n target genes to add negative edges to
                negative_edge_idxs = random.choices(range(num_genes_per_genome), k = num_negative_edges_per_gene_lst[ribap_group_count])
                heterolog_scores = simulate_bit_scores(200, 10000, len(negative_edge_idxs))

                for negative_edge_idx, score in zip(negative_edge_idxs, heterolog_scores):

                    # add heterolog sim scores to sim score dict both ways
                    negative_target = gene_lsts[target_genome_idx][negative_edge_idx]
                    similarity_dict[source][negative_target] = score
                    similarity_dict[negative_target][source] = score
                    neg_edge_count += 2

    if (nested_len(similarity_dict) / (num_total_edges * 2)) < 0.8: log.warning(f'Number of similarity scores in dictionary ({nested_len(similarity_dict)}) diverges by more than 20 % ({(nested_len(similarity_dict) / (num_total_edges * 2)) *100:.2f} %) from number of expected similarity edges ({num_total_edges * 2}).')
    else: log.info(f'Generated {(nested_len(similarity_dict) / (num_total_edges * 2)) * 100} % ({nested_len(similarity_dict)}) of expected edges ({num_total_edges * 2}), which is in within the tolerated variance (80 %).')
    assert len(ribap_groups_dict) == (num_genes_per_genome * (num_genomes)) , f'Number of ribap group mappings found ({len(ribap_groups_dict)}) is not equal to number of expected ribap group mappings ({num_genes_per_genome * (num_genomes)}).'
    assert len(ribap_groups_lst) == num_genes_per_genome , f'Number of ribap groups ({len(ribap_groups_lst)}) is not equal to number of genes per genome ({num_genes_per_genome}), but every gene should belong to one RIBAP group.'
    return similarity_dict, ribap_groups_dict, ribap_groups_lst


def shuffle_synteny_blocks(genomes_lst, k, n):
    """
    Fragment each genome into synteny blocks of size k of which n blocks are shuffled within the genome.
    """

    if n <= 1:
        log.info('Cannot shuffle a single synteny block but blocks to shuffle is set < 1, returning original gene synteny..')
        return genomes_lst

    log.info('Generating and shuffling gene synteny blocks..')

    shuffled_genomes = [None] * len(genomes_lst)
    genome_idx = 0

    for genome in genomes_lst:
        genome_fragments = list(chunks(genome, k))
        fragments_to_shuffle_indices = random.choices(range(len(genome_fragments)), k = n)

        selected_frags = [genome_fragments[i] for i in fragments_to_shuffle_indices]
        random.shuffle(selected_frags)

        for idx, shuffled_value in zip(fragments_to_shuffle_indices, selected_frags):
            genome_fragments[idx] = shuffled_value
        
        shuffled_genomes[genome_idx] = [x for xs in genome_fragments for x in xs]

    return shuffled_genomes
