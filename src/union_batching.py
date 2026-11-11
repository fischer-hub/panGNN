"""Optional batching (--union_batches): one sub graph per batch, cut out of the whole graph during training.

Instead of building every group's sub graph up front (generate_sub_graphs), the similarity edges of all genes
are stored once as a CSR (edges sorted by source gene). A batch of RIBAP groups is cut out the same way
generate_sub_graphs builds a single sub graph: n similarity hops from the group genes, then their +-n genomic
neighbours (global gene positions, as get_neighbour_graph), then all similarity edges among these genes.
Genes shared by several groups of a batch appear only once.
"""
import random
import torch
from torch_geometric.data import Data
from src.preprocessing import TRAIN, NO_SPLIT


class WholeGraph:
    """Similarity edges of all genes as a CSR, plus reusable scratch arrays for cutting out batches."""

    def __init__(self, gene_ids, sim_score_dict, ribap_groups_lst, gene_split, n_hops):
        self.gene_ids = gene_ids
        self.n_hops = n_hops
        self.node_of = {gene: idx for idx, gene in enumerate(gene_ids)}
        n_nodes = len(gene_ids)

        sources, targets, weights = [], [], []
        for query, candidates in sim_score_dict.items():
            for target, score in candidates.items():
                if target == query or target not in self.node_of: continue
                sources.append(self.node_of[query])
                targets.append(self.node_of[target])
                weights.append(score)

        sources, targets = torch.tensor(sources, dtype = torch.long), torch.tensor(targets, dtype = torch.long)
        by_source = torch.argsort(sources, stable = True)
        self.sources, self.targets = sources[by_source], targets[by_source]
        self.weights = torch.tensor(weights, dtype = torch.float32)[by_source]
        self.row_ptr = torch.zeros(n_nodes + 1, dtype = torch.long)
        self.row_ptr[1:] = torch.cumsum(torch.bincount(self.sources, minlength = n_nodes), 0)

        group_of_node = torch.full((n_nodes,), -1, dtype = torch.long)
        for group_idx, group in enumerate(ribap_groups_lst):
            group_of_node[[self.node_of[gene] for gene in group if gene in self.node_of]] = group_idx
        source_group, target_group = group_of_node[self.sources], group_of_node[self.targets]
        self.labels = ((source_group >= 0) & (source_group == target_group)).float()
        self.split_of_node = torch.tensor([gene_split.get(gene, NO_SPLIT) for gene in gene_ids])

        self.in_set = torch.zeros(n_nodes, dtype = torch.bool)
        self.local_of = torch.full((n_nodes,), -1, dtype = torch.long)
        self.offsets = torch.tensor([offset for offset in range(-n_hops, n_hops + 1) if offset != 0])


    def edges_of(self, nodes):
        """Ids of all CSR edges leaving these nodes, in one vectorised gather."""
        starts = self.row_ptr[nodes]
        counts = self.row_ptr[nodes + 1] - starts
        first_of_row = torch.cumsum(counts, 0) - counts
        return torch.arange(int(counts.sum())) - torch.repeat_interleave(first_of_row, counts) + torch.repeat_interleave(starts, counts)


    def cut(self, seed_nodes):
        """Genes, similarity edge ids and neighbour edges of the union sub graph around the seed genes."""
        frontier = torch.unique(seed_nodes)
        self.in_set[frontier] = True

        for _ in range(self.n_hops):
            reached = self.targets[self.edges_of(frontier)]
            frontier = torch.unique(reached[~self.in_set[reached]])
            self.in_set[frontier] = True

        hop_nodes = self.in_set.nonzero().squeeze(1)
        self.in_set[hop_nodes] = False

        neighbour_pos = hop_nodes[:, None] + self.offsets[None, :]
        valid = (neighbour_pos >= 0) & (neighbour_pos < len(self.gene_ids))
        nodes = torch.unique(torch.cat([hop_nodes, neighbour_pos[valid]]))
        self.local_of[nodes] = torch.arange(len(nodes))

        edge_ids = self.edges_of(nodes)
        edge_ids = edge_ids[self.local_of[self.targets[edge_ids]] >= 0]
        sim_edge_index = torch.stack([self.local_of[self.sources[edge_ids]], self.local_of[self.targets[edge_ids]]])

        # neighbour edges as in get_neighbour_graph: from the hop genes to their neighbours, both directions
        origins, neighbours = hop_nodes[:, None].expand_as(neighbour_pos)[valid], neighbour_pos[valid]
        pairs = torch.cat([torch.stack([origins, neighbours]), torch.stack([neighbours, origins])], dim = 1)
        neighbour_edge_index = self.local_of[torch.unique(pairs, dim = 1)]

        self.local_of[nodes] = -1
        return nodes, edge_ids, sim_edge_index, neighbour_edge_index


    def batch_graph(self, seed_nodes, split, node_features, keep_gene_ids = False):
        """Union sub graph around the seed genes, edges with both genes in split are supervised."""
        nodes, edge_ids, sim_edge_index, neighbour_edge_index = self.cut(seed_nodes)
        gene_lst = [self.gene_ids[node] for node in nodes.tolist()]

        graph = Data(node_features(gene_lst), sim_edge_index, self.weights[edge_ids], self.labels[edge_ids])
        graph.neighbour_edge_index = neighbour_edge_index
        graph.target_mask = (self.split_of_node[self.sources[edge_ids]] == split) & (self.split_of_node[self.targets[edge_ids]] == split)
        if keep_gene_ids: graph.gene_lst = gene_lst

        return graph


    def full_graph(self, split, node_features):
        """The whole graph (every gene is its own seed), edges with both genes in split are scored."""
        return self.batch_graph(torch.arange(len(self.gene_ids)), split, node_features, keep_gene_ids = True)


    def class_balance(self, split = TRAIN):
        """Negative / positive edges among all edges of a split, every edge counted once."""
        in_split = (self.split_of_node[self.sources] == split) & (self.split_of_node[self.targets] == split)
        n_positive = self.labels[in_split].sum()
        return float((in_split.sum() - n_positive) / n_positive)


def batches_in_unit_order(units, batch_size):
    """Concatenate the groups unit after unit and cut them into batches, so groups of a split unit stay together."""
    groups = [group for unit in units for group in unit]
    return [torch.cat(groups[start:start + batch_size]) for start in range(0, len(groups), batch_size)]


class UnionBatchDataset(torch.utils.data.Dataset):
    """Training batches, one item is one union sub graph. The split units are reshuffled every epoch."""

    def __init__(self, whole_graph, units, split, batch_size, node_features, seed):
        self.whole_graph, self.units, self.split = whole_graph, units, split
        self.batch_size, self.node_features, self.seed = batch_size, node_features, seed
        self.reshuffle(0)


    def reshuffle(self, epoch):
        units = list(self.units)
        random.Random(self.seed + epoch).shuffle(units)
        self.batches = batches_in_unit_order(units, self.batch_size)


    def __len__(self):
        return len(self.batches)


    def __getitem__(self, idx):
        return self.whole_graph.batch_graph(self.batches[idx], self.split, self.node_features)
