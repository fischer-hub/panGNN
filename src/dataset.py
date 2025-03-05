from src.preprocessing import load_gff, load_similarity_score, load_ribap_groups, build_edge_index, map_edge_weights, map_labels_to_edge_index, construct_neighbour_lst, generate_neighbour_edge_features, build_adjacency_vectors, normalize_sim_scores
from src.setup import log, args
from src.helper import concat_graph_data, simulate_dataset, generate_minimal_dataset, sub_sample_graph_edges
import torch, os, pickle, random
from torch_geometric.data import Dataset, Data
from rich.progress import track, Console, Progress
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.transforms import RemoveDuplicatedEdges
from src.plot import plot_violin_distributions, plot_homolog_positions


class HomogenousDataset(Dataset):
    """Class holding the input graph datastructures.

    First call instructor of the class to load data from the input files.

    If you want to scale the edge weights by their neimghbourhood similarity scores this is the right time to call scale_edge__weights().

    Now generate the respective Data() objects for each subgraph in the input data with generate_data().

    If you want to concat the neigbourhood similarity score edge weights and the normal sim score edge weights, call concate_edge_weights() now.

    Split the data points into train, test and validation sets using split_data().
    """
    def __init__(self, gff_files = [], similarity_score_file = '', ribap_groups_file = None, num_neighbours = 1):
        super().__init__(root = None, transform = None, pre_transform = None, pre_filter = None)

        self.gff_files = gff_files
        self.similarity_score_file = similarity_score_file
        self.num_neighbours = num_neighbours
        self.ribap_groups_file = ribap_groups_file
        self.data_lst = []
        self.gene_str_ids_lst = []
        self.neighbour_lst = []
        self.train = []
        self.test = []
        self.val = []

    def sub_sample_graph_edges(self, fraction = 0.8):
        """Subsample graph by sampling from the edge, weight, and label tensors, effectively removing 1-fraction edges from the resulting graph.

        Args:
            fraction (float, optional): Fraction of edges and labels to sample for the resulting graph. Defaults to 0.8.

        Returns:
            PyG Data object: The randomly subsampled graph
        """

        indices = random.sample(range(0, len(self.labels_ts)), int(len(self.labels_ts) * fraction))

        #gene_str_ids_lst = [self.gene_str_ids_lst[i] for i in component_nodes]
        #component_normalized_gene_positions_ts = torch.tensor([normalized_gene_positions_ts[i] for i in component_nodes])

        # remap nodes too?

        edge_index_origin = torch.index_select(self.edge_index_ts[0], 0, torch.tensor(indices))
        edge_index_target = torch.index_select(self.edge_index_ts[1], 0, torch.tensor(indices))
        edge_index = torch.stack((edge_index_origin, edge_index_target))
        edge_weights = torch.index_select(self.edge_weight_ts, 0, torch.tensor(indices))
        labels = torch.index_select(self.labels_ts, 0, torch.tensor(indices))

        return Data(self.node_features_ts, edge_index, edge_weights, labels)
    

    def generate_graph_data(self, called_from_child_class = False):
        """Generate datastructures that represent the input graph from the input data."""

        genome_annotation_df_lst = []
        genome_name_lst = []
        num_genes = 0
        self.data_lst = []
        self.gene_id_position_dict = {}

        # load annotations from gff files and format to pandas dataframe
        for gff_file in track(self.gff_files, description='Loading annotation files..', transient=True):
            genome_annotation_df = load_gff(gff_file)

            if 'hemB' not in genome_annotation_df.iloc[0].values[-1]:
                log.error('Annotation data does not start with start gene, uncentered gene data will lead to falsy gene positions.')
                log.error(f"Annotation data starts with '{genome_annotation_df.iloc[0].values}'")

            genome_annotation_df_lst.append(genome_annotation_df)
            log.info(f"Loaded annotation file of genome number {len(genome_annotation_df_lst)}: {gff_file}")
            log.debug(f"Genome 1 annotation dataframe:\n {genome_annotation_df}")
            num_genes += len(genome_annotation_df.index)
            
            self.gene_str_ids_lst += list(genome_annotation_df.index)
            
            # for each string gene ID save its normalized position in the gff file into the dictionary
            self.gene_id_position_dict.update({gene: (idx / len(list(genome_annotation_df.index))) for idx, gene in enumerate(list(genome_annotation_df.index))})


            genome_name_lst.append(os.path.basename(gff_file).split('.')[0].replace('_RENAMED', ''))

        #self.neighbour_lst = construct_neighbour_lst(num_genes, self.num_neighbours)

        # total number of genes found in all annotation files
        log.info(f"Total number of genes found in annotation files: {num_genes}")
        self.gene_id_integer_dict = {gene: idx for idx, gene in enumerate(self.gene_str_ids_lst)}
        self.gene_ids_ts = torch.tensor(list(self.gene_id_integer_dict.values()))
        normalized_gene_positions_ts = torch.tensor([1 for pos in list(self.gene_id_position_dict.values())]).float()#.unsqueeze(1)
        self.node_features_ts = normalized_gene_positions_ts.unsqueeze(1)

        # load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
        sim_score_dict = load_similarity_score(self.similarity_score_file)

        with Console().status("Building edge index..") as status:
            self.edge_index_ts = build_edge_index(sim_score_dict, self.gene_id_integer_dict, fully_connected = False)
        log.info('Successfully built edge index')
        
        with Console().status("Mapping edge weights to respective edge index positions..") as status:
            self.edge_weight_ts = map_edge_weights(self.edge_index_ts, sim_score_dict, self.gene_str_ids_lst, use_cache=True) #torch.randn((num_genes/2, edge_feature_dim))  # Edge 
        log.info('Successfully mapped weights to the edge index')

        #self.neighbour_edge_weights_ts = generate_neighbour_edge_features(self.neighbour_lst, self.edge_index_ts, sim_score_dict, self.gene_str_ids_lst)
        self.neighbour_edge_weights_ts = None
        
        #self.adjacency_vectors_ts = build_adjacency_vectors(num_neighbours = 2, gene_id_lst = self.gene_ids_ts)


        if self.ribap_groups_file:

            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict = load_ribap_groups(self.ribap_groups_file, genome_name_lst)

            # construct list of labels from ribap groups and format to match edge_index
            with Console().status("Mapping labels to gene pairs in edge index.") as status:
                self.labels_ts = map_labels_to_edge_index(self.edge_index_ts, self.gene_str_ids_lst, self.ribap_groups_dict, use_cache=True)
            log.info(f"{self.labels_ts.sum().item() / len(self.labels_ts) * 100} % of labels are in positive class.")
            self.class_balance = (self.labels_ts == 0.).sum()/self.labels_ts.sum()
            log.info('Successfully mapped labels to gene pairs in edge index')
        else:
            self.labels_ts = None
            self.class_balance = None

        if not called_from_child_class:
            self.test = self.sub_sample_graph_edges()
            self.train = [self.sub_sample_graph_edges()]

        return


    
    def split_data(self, split = (0.7, 0.15, 0.15), batch_size = 32):
        """Split the singular graph data into train, test and validations sets, also create batches in the train dataset.

        Args:
            split (tuple, optional): . fraction of the graphs to go to train, test and validation sets respectively. Defaults to (0.7, 0.15, 0.15).
            batch_size (int, optional): number of graphs to trasin on in a single batch. Defaults to 32.
        """
        if not self.data_lst:
            log.error("Data object list of this dataset is empty. What have you done..")

        if args.batch_size == 1:
            log.info('Batch size set to 1, train and test datasets are the same.')
            self.train = self.data_lst
            self.test = self.data_lst[0]
            return
        
        # calculate train, test, val split and batches for train data
        num_train_data = int(len(self.data_lst) * split[0])
        num_test_data = int(len(self.data_lst) * split[1])

        random.shuffle(self.data_lst)

        log.info(f"Splitting datasets into sets of {num_train_data}, {num_test_data}, {len(self.data_lst)-(num_test_data+num_train_data)} graphs.")
        self.train = self.data_lst[:num_train_data]
        self.train = [concat_graph_data(self.train[i:i + batch_size]) for i in range(0, len(self.train), batch_size)]
        self.train = [graph for graph in self.train if graph.y.numel() > 0]
        self.test = concat_graph_data(self.data_lst[num_train_data:num_train_data + num_test_data])
        self.val = concat_graph_data(self.data_lst[num_test_data:])

    
    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
    
    def to(self, device):

        if not self.train.x:
            log.warning(f"Graph data not ready to load to device {device}, creating graph data now.")

        self.generate_graph_data()
        self.x.to(device)
        self.edge_attr.to(device)
        self.edge_index.to(device)
        self.y.to(device)
        self.data_lst.to(device)

    def scale_weights(self):
        """Scale the similarity scores on the edges of the input graph by the gene neighbourhood similarity factor
        """
        #if not self.edge_weight_ts:
        #    log.warning(f"Graph data not ready to be scaled, creating graph data now.")
        #    self.generate_graph_data()
        
        if self.data_lst:
            log.warning("You are trying to scale the edge weights but the graph data has already been generated. Please call 'generate_graph_data()' again after scaling weights to apply sclaed weights to each sub graph.")
        
        self.data_lst[0].edge_attr = self.data_lst[0].edge_attr * self.data_lst[0].neighbour_edge_weights_ts

        #self.edge_weight_ts = self.edge_weight_ts * self.neighbour_edge_weights_ts
        #self.edge_attr = self.edge_weight_ts

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self = pickle.load(f)
    
    def concate_edge_weights(self):

        for graph in self.data_lst:
            graph.edge_attr = torch.concat((graph.edge_attr, graph.neighbour_edge_weights_ts))
            graph.edge_index = torch.stack((torch.concat((graph.edge_index[0], graph.edge_index[0])), torch.concat((graph.edge_index[1], graph.edge_index[1]))))
        
        self.edge_index = torch.stack((torch.concat((self.edge_index[0], self.edge_index[0])), torch.concat((self.edge_index[1], self.edge_index[1]))))
        self.edge_weight_ts = torch.concat((self.edge_weight_ts, self.neighbour_edge_weights_ts))
        self.edge_attr = self.edge_weight_ts



class UnionGraphDataset(Dataset):
    """Class holding the input graph datastructures.

    First call instructor of the class to load data from the input files.

    If you want to scale the edge weights by their neimghbourhood similarity scores this is the right time to call scale_edge__weights().

    Now generate the respective Data() objects for each subgraph in the input data with generate_data().

    If you want to concat the neigbourhood similarity score edge weights and the normal sim score edge weights, call concate_edge_weights() now.

    Split the data points into train, test and validation sets using split_data().
    """
    def __init__(self, gff_files = [], similarity_score_file = '', ribap_groups_file = None, split = (0.7, 0.3), categorical_nodes = False):
        super().__init__(root = None, transform = None, pre_transform = None, pre_filter = None)

        genome_annotation_df_lst = []
        genome_name_lst = []
        self.gene_str_ids_lst_train = []
        self.gene_str_ids_lst_val = []
        self.gene_str_int_lst_train = []
        self.gene_str_int_lst_val = []
        self.gene_id_position_dict = {}

        self.categorical_nodes = categorical_nodes
        num_genes = 0
        self.split = split
        self.val = torch.tensor([])

        
        if not gff_files:
            if not args.simulate_dataset:
                log.info("No annotation files provided, use generate_minimal_dataset() or simulate_dataset() to generate graph data for this object.")
            return
        
        # load annotations from gff files and format to pandas dataframe
        for gff_file in track(gff_files, description='Loading annotation files..', transient=True):
            genome_annotation_df = load_gff(gff_file)

            if 'hemB' not in genome_annotation_df.iloc[0].values[-1]:
                log.error('Annotation data does not start with start gene, uncentered gene data will lead to falsy gene positions.')
                log.error(f"Annotation data starts with '{genome_annotation_df.iloc[0].values}'")

            genome_annotation_df_lst.append(genome_annotation_df)
            log.info(f"Loaded annotation file of genome number {len(genome_annotation_df_lst)}: {gff_file}")
            log.debug(f"Genome 1 annotation dataframe:\n {genome_annotation_df}")
            num_genes += len(genome_annotation_df.index)
            
            self.gene_str_ids_lst_train += list(genome_annotation_df.index)[:int(len(genome_annotation_df.index) * split[0])]
            self.gene_str_ids_lst_val   += list(genome_annotation_df.index)[int(len(genome_annotation_df.index) * split[0]):]

            self.gene_str_int_lst_train += range(len(self.gene_str_ids_lst_train))
            self.gene_str_int_lst_val   += range(len(self.gene_str_ids_lst_val))


            #self.gene_str_ids_lst_test  += list(genome_annotation_df.index)[]
            
            # for each string gene ID save its normalized position in the gff file into the dictionary
            self.gene_id_position_dict.update({gene: idx for idx, gene in enumerate(list(genome_annotation_df.index))})

            genome_name_lst.append(os.path.basename(gff_file).split('.')[0].replace('_RENAMED', ''))

        log.info(f"Total number of genes found in annotation files: {num_genes}")
        #self.neighbour_lst = construct_neighbour_lst(num_genes, self.num_neighbours)
        
        self.sim_score_dict = load_similarity_score(similarity_score_file)

        prob_lst = []
        qscore_lst = []

        #for temp in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5, 10]:
        #    prob_lst.append((temp, normalize_sim_scores(self.sim_score_dict, t = temp, pseudo_count = 1, q_score_norm= False)))
        #    qscore_lst.append((temp, normalize_sim_scores(self.sim_score_dict, t = temp, pseudo_count = 1, q_score_norm=True)))



        self.sim_score_dict = normalize_sim_scores(self.sim_score_dict, t = args.normalization_temp, pseudo_count = 1, q_score_norm=args.no_q_score_transform)

        if ribap_groups_file:
            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict = load_ribap_groups(ribap_groups_file, genome_name_lst)
            plot_homolog_positions(self.ribap_groups_dict, self.gene_id_position_dict)
        else:
            self.ribap_groups_dict = None
            self.labels_ts = None
            self.class_balance = None

        #plot_violin_distributions(prob_lst, self.ribap_groups_dict, prob = True, path = os.path.join('plots', 'normalized_scores_violin_prob.png'))
        #plot_violin_distributions(qscore_lst, self.ribap_groups_dict, prob = False, path = os.path.join('plots', 'normalized_scores_violin_qscore.png'))
        #quit()
        self.train = self.generate_graphs(self.gene_str_ids_lst_train, self.gene_str_int_lst_train)
        self.test = self.generate_graphs(self.gene_str_ids_lst_val, self.gene_str_int_lst_val)

    def len(self):
        return len(self.train.x) + len(self.test.x)

    def generate_graphs(self, gene_str_ids_lst, gene_str_int_lst):
        # total number of genes found in all annotation files
        gene_id_integer_dict = {gene: idx for idx, gene in enumerate(gene_str_ids_lst)}
        gene_ids_ts = torch.tensor(list(gene_id_integer_dict.values()))
        normalized_gene_positions_ts = torch.tensor([1 for pos in gene_ids_ts]).float()#.unsqueeze(1)
        node_features_ts = normalized_gene_positions_ts.unsqueeze(1)


        # load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe

        with Console().status("Building edge index..") as status:
            edge_index_ts = build_edge_index(self.sim_score_dict, gene_id_integer_dict, fully_connected = False)
        log.info('Successfully built edge index')

        with Console().status("Mapping edge weights to respective edge index positions..") as status:
            edge_weight_ts = map_edge_weights(edge_index_ts, self.sim_score_dict, gene_str_ids_lst, use_cache=False) #torch.randn((num_genes/2, edge_feature_dim))  # Edge 
        log.info('Successfully mapped weights to the edge index')

        # construct list of labels from ribap groups and format to match edge_index
        with Console().status("Mapping labels to gene pairs in edge index.") as status:
            labels_ts = map_labels_to_edge_index(edge_index_ts, gene_str_ids_lst, self.ribap_groups_dict, use_cache=False) if self.ribap_groups_dict else None
        log.info(f"{labels_ts.sum().item() / len(labels_ts) * 100} % of labels are in positive class.")
        self.class_balance = (labels_ts == 0.).sum()/labels_ts.sum()
        log.info('Successfully mapped labels to gene pairs in edge index')
        

        origin_idx, target_idx = [], []
        # add edges to n nearest neighbour nodes
        for gene_id in track(gene_id_integer_dict.values(), description = 'Adding edges to neighbouring nodes..', transient = True):
            for neighbour_id in range(gene_id - args.neighbours, gene_id + args.neighbours + 1):
                if neighbour_id in gene_id_integer_dict.values():
                    origin_idx.append(gene_id)
                    # we anyway add the neighbour edges in both directions when iterating over both nodes
                    #origin_idx.append(neighbour_id)
                    #target_idx.append(gene_id)
                    target_idx.append(neighbour_id)

        union_edge_index_ts = torch.stack((torch.cat((edge_index_ts[0], torch.tensor(origin_idx))), torch.cat((edge_index_ts[1], torch.tensor(target_idx)))))

        transform = RemoveDuplicatedEdges()
        union_edge_index_ts = transform(Data(x = normalized_gene_positions_ts, edge_index = union_edge_index_ts, edge_attr = None, y = None)).edge_index

        if self.categorical_nodes:
            x = torch.tensor(gene_str_int_lst)
        else:
            x = normalized_gene_positions_ts.unsqueeze(1)

        graph_data = Data(x, edge_index_ts, edge_weight_ts, labels_ts)
        graph_data.union_edge_index = union_edge_index_ts
        # this makes the Data object class crash on print??
        #graph_data.subsample_graph_edges = sub_sample_graph_edges.__get__(graph_data, Data)

        return graph_data
    
    # TODO: this function is moved to src.helper right??
    def sub_sample_graph_edges(self, graph, fraction = 0.8, sample_pos_edges = False):
        """Subsample graph by sampling from the edge, weight, and label tensors, effectively removing 1-fraction edges from the resulting graph.

        Args:
            fraction (float, optional): Fraction of edges and labels to sample for the resulting graph. Defaults to 0.8.

        Returns:
            PyG Data object: The randomly subsampled graph
        """
        num_neighbour_edges = len(graph.union_edge_index[0]) - len(graph.y)
        
        if sample_pos_edges:
            
            sim_indices = random.sample(range(0, len(graph.y)), int(len(graph.y) * fraction))
            sim_labels = torch.index_select(graph.y, 0, torch.tensor(sim_indices))
        
        # dont leave pos edges behind, sample the counter fraction from edges we leave behind but only sample from the negative indices
        # then use the inverse tensor as batch
        else:

            assert (graph.y.sum() / len(graph.y)) <= fraction, f'Trying to subsample {fraction} of the edges in the data when only {graph.y.sum() / len(graph.y)} of edges are positive is not possible when sample_pos_edges is set to False, increase positive edges in data or decrease subsampling fraction'

            negative_labels_indices = torch.nonzero(graph.y == 0, as_tuple=True)[0]
            counter_frac_sim_indices = random.sample(negative_labels_indices, int(len(graph.y) * (1 - fraction)))
            all_indices = torch.arange(len(graph.y))
            inverse_indices = all_indices[~torch.isin(all_indices, counter_frac_sim_indices)]
            sim_labels = torch.index_select(graph.y, 0, torch.tensor(inverse_indices))
            print('pos edges: ', graph.y.sum())
            
            assert (graph.y.sum() - sim_labels.sum()) <= 2, f'A total of {graph.y.sum() - sim_labels.sum()} positive edges have been removed during the graph subsampling but at most 2 are allowed when sample_pos_edges is set to False.'

        # maybe we should sample from the additional edges in the union, such that the edges in the sim part are the same in every batch?
        union_indices = sim_indices + random.sample(range(len(graph.y), len(graph.y) + num_neighbour_edges), int(num_neighbour_edges * fraction))

        # remap nodes too?
        sim_edge_index_origin = torch.index_select(graph.edge_index[0], 0, torch.tensor(sim_indices))
        sim_edge_index_target = torch.index_select(graph.edge_index[1], 0, torch.tensor(sim_indices))
        union_edge_index_origin = torch.index_select(graph.union_edge_index[0], 0, torch.tensor(union_indices))
        union_edge_index_target = torch.index_select(graph.union_edge_index[1], 0, torch.tensor(union_indices))
        sim_edge_index = torch.stack((sim_edge_index_origin, sim_edge_index_target))
        union_edge_index = torch.stack((union_edge_index_origin, union_edge_index_target))
        sim_edge_weights = torch.index_select(graph.edge_attr, 0, torch.tensor(sim_indices))
        #union_edge_weights = torch.index_select(self.union_edge_weight_ts, 0, torch.tensor(union_indices))
        #union_labels = torch.index_select(self.labels_ts, 0, torch.tensor(union_indices))
        
        graph = Data(graph.x, sim_edge_index, sim_edge_weights.float(), sim_labels)
        graph.union_edge_index = union_edge_index

        return graph
    
    def simulate_dataset(self, num_genes, num_genomes, class_balance = 0.2):
        num_train_genes = int(num_genes)
        num_test_genes = int(num_genes * self.split[1])
        self.train = simulate_dataset(num_train_genes, num_genomes, class_balance)
        self.test = simulate_dataset(num_test_genes, num_genomes, class_balance)


    def generate_minimal_dataset(self):
        log.info("Subsampling is not recommended on the minimal dataset, since this can easily unbalance the data.")
        self.train = generate_minimal_dataset()
        self.test = generate_minimal_dataset()


    def graph_to(self, graph, device):
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.y = graph.y.to(device)
        graph.union_edge_index = graph.union_edge_index.to(device)

        return graph


    def to(self, device):
        self.train = self.train.to(device)
        self.test = self.test.to(device)
        self.val = self.val.to(device)


    def save(self, path):

        save_dict = {}

        if os.path.exists(path):
            log.warning('')
        
        if self.train:
            save_dict['train'] = self.train.to_dict()
        else:
            log.info("Did not find any training data in the dataset to be save.")

        if self.test:
            save_dict['test'] = self.test.to_dict()
        else:
            log.info("Did not find any test data in the dataset to be save.")

        if save_dict:
            with open(path, 'wb') as handle:
                pickle.dump(save_dict, handle, protocol = -1)
        else:
            log.error("The dataset you try to save does not contain any data.")
            quit()


    def load(self, path):

        if os.path.exists(path):
            log.info(f'Loading pickled dataset from file {path}..')
            with open(path, 'rb') as handle:
                load_dict = pickle.load(handle)
        else:
            log.info(f'File {path} doesn\'t exists, exiting.')
            quit()        

        if 'train' in load_dict:
            self.train = Data().from_dict(load_dict['train'])
        else:
            log.info("Did not find any training data in the dataset that is being loaded.")
            self.train = None
                
        if 'test' in load_dict:
            self.test = Data().from_dict(load_dict['test'])
        else:
            log.info("Did not find any test data in the dataset that is being loaded.")
            self.test = None

        if not self.test and not self.train:
            log.error("Failed to load dataset from file, check that file contains at least one of training or test data.")
            quit()

        



