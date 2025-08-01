import torch, os, pickle, random, time, math
from src.preprocessing import load_gff, load_similarity_score, load_ribap_groups, build_edge_index, map_edge_weights, map_labels_to_edge_index, construct_neighbour_lst, generate_neighbour_edge_features, build_adjacency_vectors, normalize_sim_scores
from src.setup import log, args
from src.helper import concat_graph_data, simulate_dataset, generate_minimal_dataset, calculate_baseline_labels, get_connected_nodes, get_neighbour_graph, remove_duplicate_edges_tuple, char_id_generator, format_duration
from torch_geometric.data import Dataset, Data
from rich.progress import track, Console, Progress
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.transforms import RemoveDuplicatedEdges
from src.plot import plot_violin_distributions, plot_homolog_positions, plot_simscore_distribution_by_class
from multiprocessing import Pool, current_process
from src.simulate import simulate_gene_ids, simulate_similarity_scores_and_ribap_dict, shuffle_synteny_blocks

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


            genome_name_lst.append(os.path.basename(gff_file).rsplit('.', 1)[0].replace('_RENAMED', ''))

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
    def __init__(self, gff_files = [], similarity_score_file = '', ribap_groups_file = None, split = (0.7, 0.15, 0.15), categorical_nodes = False, calculate_baseline = False):
        super().__init__(root = None, transform = None, pre_transform = None, pre_filter = None)

        genome_annotation_df_lst = []
        genome_annotation_df = None
        genome_name_lst = []
        self.gene_str_ids_lst_train = []
        self.gene_str_ids_lst_val = []
        self.gene_str_ids_lst = []
        self.gene_id_position_dict = {}

        self.data_lst = []
        self.base_labels = []

        self.categorical_nodes = categorical_nodes
        self.num_genes = 0
        self.split = split
        self.val = []
        self.train = []
        self.test = []
        self.class_balance = None

        self.gff_is_subset = False
        self.calculate_baseline = calculate_baseline

        if calculate_baseline:
            log.info('Baseline calculation is set to True, note that this can slow down preprocessing and metric calculation!')

        
        if not gff_files or args.simulate_dataset:
            if not args.simulate_dataset:
                log.info("No annotation files provided, use generate_minimal_dataset() or simulate_dataset() to generate graph data for this object.")
                return
            else:
                # num_genes_per_genome, num_genomes, fraction_pos_edges, num_fragments, num_frags_to_shuffle
                num_genes_per_genome, num_genomes, frac_pos_edges, num_fragments, num_frags_to_shuffle = args.simulate_dataset
                log.info(f'Simulating dataset with input parameters: {args.simulate_dataset}')
                frag_size = math.floor(num_genes_per_genome / num_fragments)
                self.num_genes = num_genes_per_genome * num_genomes
                self.gene_str_ids_lst_old, gene_id_by_genome_lst = simulate_gene_ids(num_genes_per_genome, num_genomes)
                
                # where do we use the shuffled gene synteny order???
                self.sim_score_dict_raw, self.ribap_groups_dict, self.ribap_groups_lst = simulate_similarity_scores_and_ribap_dict(gene_id_by_genome_lst, frac_pos_edges)
                self.gene_id_by_genome_lst = shuffle_synteny_blocks(gene_id_by_genome_lst, k = frag_size, n = int(num_frags_to_shuffle))
                self.gene_str_ids_lst = [x for xs in self.gene_id_by_genome_lst for x in xs]
                self.gene_id_position_dict = {gene: idx for idx, gene in enumerate(self.gene_str_ids_lst)}
        else:
        
            # load annotations from gff files and format to pandas dataframe
            for file_counter, gff_file in track(enumerate(gff_files), description='Loading annotation files..', transient=True):
                genome_annotation_df = load_gff(gff_file)

                if 'hemB' not in genome_annotation_df.iloc[0].values[-1]:
                    log.error('Annotation data does not start with start gene, uncentered gene data will lead to falsy gene positions.')
                    log.error(f"Annotation data starts with '{genome_annotation_df.iloc[0].values}'")

                log.info(f"Loaded annotation file of genome number {file_counter + 1}: {gff_file}")
                log.debug(f"Genome 1 annotation dataframe:\n {genome_annotation_df}")
                self.num_genes += len(genome_annotation_df.index)
                
                #self.gene_str_ids_lst_train += list(genome_annotation_df.index)[:int(len(genome_annotation_df.index) * split[0])]
                #self.gene_str_ids_lst_val   += list(genome_annotation_df.index)[int(len(genome_annotation_df.index) * split[0]):]
                self.gene_str_ids_lst       += list(genome_annotation_df.index)

                #self.gene_str_ids_lst_test  += list(genome_annotation_df.index)[]
                
                # for each string gene ID save its position in the gff file into the dictionary

                genome_name_lst.append(os.path.basename(gff_file).rsplit('.', 1)[0].replace('_RENAMED', ''))

            log.info(f"Total number of genes found in annotation files: {self.num_genes}")
            #self.neighbour_lst = construct_neighbour_lst(num_genes, self.num_neighbours)
            
            self.gene_id_position_dict = {gene: idx for idx, gene in enumerate(self.gene_str_ids_lst)}

            self.sim_score_dict_raw = load_similarity_score(similarity_score_file, self.gene_id_position_dict)

        #for temp in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5, 10]:
        #    prob_lst.append((temp, normalize_sim_scores(self.sim_score_dict, t = temp, pseudo_count = 1, q_score_norm= False)))
        #    qscore_lst.append((temp, normalize_sim_scores(self.sim_score_dict, t = temp, pseudo_count = 1, q_score_norm=True)))



        if args.normalization_temp != 0:
            self.sim_score_dict = normalize_sim_scores(self.sim_score_dict_raw, t = args.normalization_temp, pseudo_count = 1, q_score_norm=args.no_q_score_transform)
        else:
            log.warning("Similarity score normalization temp set to 0, skipping normalization. This can decrease model performance.")

        if ribap_groups_file:
            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict, self.ribap_groups_lst, self.gff_is_subset = load_ribap_groups(ribap_groups_file, genome_name_lst)
            #plot_homolog_positions(self.ribap_groups_dict, self.gene_id_position_dict)
        else:
            if not args.simulate_dataset: self.ribap_groups_dict = None
            self.labels_ts = None
            self.class_balance = None

        #plot_violin_distributions(prob_lst, self.ribap_groups_dict, prob = True, path = os.path.join('plots', 'normalized_scores_violin_prob.png'))
        #plot_violin_distributions(qscore_lst, self.ribap_groups_dict, prob = False, path = os.path.join('plots', 'normalized_scores_violin_qscore.png'))
        #quit()
        ribap_groups_chunked = [self.ribap_groups_lst[i::args.cpus] for i in range(args.cpus) if self.ribap_groups_lst[i::args.cpus]]
        
        # free some mem before multiprocessing and copying old objects
        del genome_annotation_df
        del genome_name_lst
        del self.ribap_groups_lst
        del genome_annotation_df_lst


        if args.train:
            mstart = time.time()

            with Console().status("Generating sub-graphs (this might take some time)..") as status, Pool(processes = args.cpus) as pool:
                results  = pool.map(self.generate_sub_graphs, ribap_groups_chunked)
                self.data_lst = [sublist for tup in results for sublist in tup[0]]
                if self.calculate_baseline: 
                    self.base_labels = [sublist for tup in results for sublist in tup[2]]
                    self.base_labels_raw = [sublist for tup in results for sublist in tup[3]]
                class_balance_lst = [result[1] for result in results]
                self.class_balance = sum(class_balance_lst) / len(class_balance_lst)
            
            mend = time.time()
            log.info(f'Generated sub-graphs successfully, elapsed time: {format_duration(mend-mstart)}.')

            del results
            del class_balance_lst
            
            self.split_data(split, args.batch_size)

            if args.simulate_dataset:
                self.test = [self.generate_graphs()]

            if args.to_pickle:
                self.save(args.to_pickle)
        
        else:
            log.info("PanGNN not in training mode, skipping graph batching because 'ground truth' labels necessary for splitting are unknown in inference mode.")

            self.test = [self.generate_graphs()]

    def len(self):
        return len(self.train.x) + len(self.test.x)
    
    # pygs dataloader does this on its own im going to cry, remider to read the docs before reinventing the wheel
    def split_data(self, split = (0.7, 0.15, 0.05), batch_size = 32):
        """Split the singular graph data into train, test and validations sets, also create batches in the train dataset.

        Args:
            split (tuple, optional): . fraction of the graphs to go to train, test and validation sets respectively. Defaults to (0.7, 0.15, 0.15).
            batch_size (int, optional): number of graphs to trasin on in a single batch. Defaults to 32.
        """
        if not self.data_lst:
            log.error("Data object list of this dataset is empty. What have you done..")

        # calculate train, test, val split and batches for train data
        num_train_data = int(len(self.data_lst) * split[0])
        num_val_data = int(len(self.data_lst) * split[1])
        num_test_data = int(len(self.data_lst) * split[2])

        
        if self.base_labels:
            tmp = list(zip(self.data_lst, self.base_labels, self.base_labels_raw))
            random.shuffle(tmp)
            self.data_lst, self.base_labels, self.base_labels_raw = zip(*tmp)
        else:
            random.shuffle(self.data_lst)

        log.info(f"Splitting data ({len(self.data_lst)}) into sets of train: {num_train_data}, test: {num_test_data}, val: {num_val_data} graphs.")
        
        if self.calculate_baseline:

            for graph in self.data_lst[:num_train_data]:
                del graph.gene_lst
                self.train.append(graph)

            for graph in self.data_lst[num_train_data:num_train_data + num_val_data]:
                del graph.gene_lst
                self.val.append(graph)
            
            self.test  = self.data_lst[-num_test_data:]
            self.base_labels = [elem for sublist in self.base_labels[-num_test_data:] for elem in sublist]
            self.base_labels_raw = [elem for sublist in self.base_labels_raw[-num_test_data:] for elem in sublist]
        else:
            self.train = self.data_lst[:num_train_data]
            self.val   = self.data_lst[num_train_data:num_train_data + num_val_data]
            self.test  = self.data_lst[-num_test_data:]
        

        #self.train = self.data_lst[:num_train_data]
        #self.train = [concat_graph_data(self.train[i:i + batch_size]) for i in range(0, len(self.train), batch_size)]
        #self.test = concat_graph_data(self.data_lst[num_train_data:num_train_data + num_test_data])
        #self.val = concat_graph_data(self.data_lst[-num_val_data:])
    

    def generate_sub_graphs(self, ribap_groups_lst):

        pos = 0
        neg = 0
        data_lst, base_labels_lst, base_labels_raw_lst = [], [], []

        for group in ribap_groups_lst:

            if len(group) <= 1:
                continue

            similar_gene_lst = get_connected_nodes(group, self.sim_score_dict, args.neighbours)
            #print(f'len before {len(similar_gene_lst)}')
            if not similar_gene_lst: continue

            assert set(group).issubset(similar_gene_lst), f'Genes from gene family {group} not part of connected similarity nodes {similar_gene_lst}.'

            neighbour_edge_index, sub_gene_id_pos_dict, gene_lst = get_neighbour_graph(similar_gene_lst, self.gene_id_position_dict, self.gene_str_ids_lst, args.neighbours)
            
            neighbour_edge_index = remove_duplicate_edges_tuple(neighbour_edge_index)
            #print(f'len after neighbour edges: {len(neighbour_edge_index[0])}')
            assert set(similar_gene_lst).issubset(gene_lst), f'Genes from similarity gene set {similar_gene_lst} not part of sub graph gene set with neighbour genes {gene_lst}.'
            assert len(neighbour_edge_index[0]) == len(neighbour_edge_index[1]), f'List or origin nodes ({len(neighbour_edge_index[0])}) is of different length than list of target nodes ({len(neighbour_edge_index[1])}), invalid edge index!'

            sub_sim_score_dict = { gene_str_id: self.sim_score_dict[gene_str_id] for gene_str_id in gene_lst if gene_str_id in self.sim_score_dict}
            
            if not sub_sim_score_dict: continue

            sim_edge_index = build_edge_index(sub_sim_score_dict, sub_gene_id_pos_dict, fully_connected = False)
            sim_edge_index = remove_duplicate_edges_tuple(sim_edge_index)
            if self.gff_is_subset and len(sim_edge_index[0]) < len(group): continue
            assert len(sim_edge_index[0]) >= len(group), f'Found less similarity edges than edges neccessary to connect genes from origin gene family, number of edges is: {len(sim_edge_index[0])}, but {len(group)} genes belong to origin gene family ({group}) of this graph.'
            assert len(sim_edge_index[0]) == len(sim_edge_index[1]), f'List or origin nodes ({len(sim_edge_index[0])}) is of different length than list of target nodes ({len(sim_edge_index[1])}), invalid edge index!'

            sim_edge_weights = map_edge_weights(sim_edge_index, sub_sim_score_dict, gene_lst, use_cache=False)
            assert len(sim_edge_weights) == len(sim_edge_index[0]), f'Number of similarity edges is different from number of edge weights (similarity scores), can not map {len(sim_edge_weights)} edge weights to {len(sim_edge_index[0])} edges.'


            if self.ribap_groups_dict or args.simulate_dataset:
                labels_ts = map_labels_to_edge_index(sim_edge_index, gene_lst, self.ribap_groups_dict, use_cache=False)
                pos += labels_ts.sum().item()
                neg += len(labels_ts) - labels_ts.sum().item()

                if self.calculate_baseline:
                    base_labels, base_labels_raw = calculate_baseline_labels(sim_edge_index, gene_lst, sub_sim_score_dict, self.sim_score_dict_raw)
                    assert len(base_labels) == len(labels_ts), f"List of normalized baseline labels ({len(base_labels)}) is not the same size as list of 'ground truth' labels ({len(labels_ts)})."
                    assert len(base_labels_raw) == len(labels_ts), f"List of raw baseline labels ({len(base_labels)}) is not the same size as list of 'ground truth' labels ({len(labels_ts)})."
                else:
                    base_labels = None
                    base_labels_raw = None

            else:
                
                base_labels_raw = None
                base_labels = None
                labels_ts = None


            x = torch.tensor([1] * len(gene_lst)).unsqueeze(1).float()

            sim_edge_index = torch.stack((
                torch.tensor((sim_edge_index[0]), dtype = torch.long),
                torch.tensor((sim_edge_index[1]), dtype = torch.long)
            ))

            if args.union_edge_weights:
                
                union_edge_index = torch.stack((
                    torch.cat((torch.tensor(neighbour_edge_index[0]), sim_edge_index[0])),
                    torch.cat((torch.tensor(neighbour_edge_index[1]), sim_edge_index[1]))
                ))


                union_edge_weights = torch.cat((
                    torch.tensor([1] * (len(union_edge_index[0]) - len(sim_edge_index[0]))),
                    sim_edge_weights
                ))

                assert len(union_edge_weights) == len(union_edge_index[0]), f'Number of similarity edges is different from number of edge weights (similarity scores), can not map {len(union_edge_weights)} edge weights to {len(union_edge_index[0])} edges.'

                graph = Data(x, sim_edge_index, union_edge_weights, labels_ts)
                graph.union_edge_index = union_edge_index.long()
            
            else:
                graph = Data(x, sim_edge_index, sim_edge_weights, labels_ts)
                graph.neighbour_edge_index = torch.stack((
                    torch.tensor((neighbour_edge_index[0]), dtype = torch.long),
                    torch.tensor((neighbour_edge_index[1]), dtype = torch.long)
                )) 

            if self.calculate_baseline:
                graph.gene_lst = gene_lst
            
            base_labels_lst.append(base_labels)
            base_labels_raw_lst.append(base_labels_raw)
            data_lst.append(graph)

        local_class_balance = neg / pos
        log.debug(f'{current_process().name} finished.')
        
        return (data_lst, local_class_balance, base_labels_lst, base_labels_raw_lst)


    def generate_graphs(self):
        # total number of genes found in all annotation files
        log.info('Allocating node feature tensor..')
        node_features_ts = torch.ones((self.num_genes,1 ), dtype = torch.float)

        with Console().status("Building edge index..") as status:
            edge_index_ts = build_edge_index(self.sim_score_dict, self.gene_id_position_dict, fully_connected = False)
            edge_index_ts = remove_duplicate_edges_tuple(edge_index_ts)
            
        log.info('Successfully built edge index')

        with Console().status("Mapping edge weights to respective edge index positions..") as status:
            edge_weight_ts = map_edge_weights(edge_index_ts, self.sim_score_dict, self.gene_str_ids_lst, use_cache=False) #torch.randn((num_genes/2, edge_feature_dim))  # Edge
        log.info('Successfully mapped weights to the edge index')

        # construct list of labels from ribap groups and format to match edge_index
        with Console().status("Mapping labels to gene pairs in edge index.") as status:
            labels_ts = map_labels_to_edge_index(edge_index_ts, self.gene_str_ids_lst, self.ribap_groups_dict, use_cache=False) if self.ribap_groups_dict else None
            
        log.info(f"{labels_ts.sum().item() / len(labels_ts) * 100} % of labels are in positive class.")
        self.class_balance = (labels_ts == 0.).sum()/labels_ts.sum()
        log.info('Successfully mapped labels to gene pairs in edge index')
        
        edge_index_ts = torch.stack((torch.tensor(edge_index_ts[0]), torch.tensor(edge_index_ts[1])))
        
        origin_idx, target_idx = [None] * (self.num_genes * (args.neighbours + 1) * 2), [None] * (self.num_genes * (args.neighbours + 1) * 2)
        edge_count = 0

        # add edges to n nearest neighbour nodes
        # NOTE: this ignores shuffled gene synteny from simulated datasets since it doesnt use the gene_id_pos_dict!!!
        for gene_id in track(range(self.num_genes), description = 'Adding edges to neighbouring nodes..', transient = True):
            for neighbour_id in range(gene_id - args.neighbours, gene_id + args.neighbours + 1):
                if neighbour_id >= 0 and neighbour_id < len(self.gene_str_ids_lst):
                    origin_idx[edge_count] = gene_id
                    target_idx[edge_count] = neighbour_id
                    edge_count += 1

        origin_idx = origin_idx[:edge_count]
        target_idx = target_idx[:edge_count]

        neighbour_edge_index_ts = torch.stack((torch.tensor(origin_idx), torch.tensor(target_idx)))

        if self.categorical_nodes:
            x = torch.ones(len(self.gene_str_ids_lst))
        else:
            x = node_features_ts
            
        if args.union_edge_weights:
            union_edge_index_ts = torch.stack((torch.cat((edge_index_ts[0], neighbour_edge_index_ts[0])), torch.cat((edge_index_ts[1], neighbour_edge_index_ts[1]))))
            assert (len(union_edge_index_ts[0]) - len(edge_index_ts[0])) > 0, f'Union edge index ({len(union_edge_index_ts[0])}) is smaller than similarity edge index ({len(edge_index_ts[0])}) but was built based on similarity edge index, something broke!'
            union_edge_weights = torch.cat((edge_weight_ts, torch.tensor([1] * (len(union_edge_index_ts[0]) - len(edge_index_ts[0])))))
            assert len(union_edge_weights) == len(union_edge_index_ts[0]), f'Number of edge weights ({len(edge_weight_ts)}) is different from number of edges in union edge index ({len(union_edge_index_ts[0])}).'
            #labels_ts      = torch.cat((labels_ts, torch.tensor([1] * (len(union_edge_index_ts[0]) - len(labels_ts)))))
        
            graph_data = Data(x, edge_index_ts, union_edge_weights, labels_ts)
            graph_data.union_edge_index = union_edge_index_ts
        else:
            graph_data = Data(x, edge_index_ts, edge_weight_ts, labels_ts)
            graph_data.neighbour_edge_index = neighbour_edge_index_ts



        # since the sub graph in inference mode is the whole graph we pass the whole sim score dict instread of the sub sim score dict here
        log.info('Calculating baseline labels for max score candidates..')
        self.base_labels, self.base_labels_raw = calculate_baseline_labels(edge_index_ts, self.gene_str_ids_lst, self.sim_score_dict, self.sim_score_dict_raw)

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
    
    # TODO: adjust this to new dataset form! implement synteny block shuffle of some kind and batching as we do on real data based around ortholog group
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
        self.train = [graph.to(device) for graph in self.train]
        self.test = self.test.to(device)
        self.val = self.val.to(device)


    def save(self, path):

        log.info(f"Saving current input dataset to file '{path}'")

        save_dict = {}

        if os.path.exists(path):
            log.warning('')
        
        for subset in ['train', 'test', 'val']:
            if getattr(self, subset):
                save_dict[subset] = [ graph.to_dict() for graph in getattr(self, subset) ]
            else:
                log.info(f"Did not find any {subset} data in the dataset to be save.")

        save_dict['class_balance'] = self.class_balance
        save_dict['num_genes'] = self.num_genes
        save_dict['base_labels'] = self.base_labels
        save_dict['base_labels_raw'] = self.base_labels_raw
        save_dict['similarity_dict'] = self.sim_score_dict

        if save_dict:
            with open(path, 'wb') as handle:
                pickle.dump(save_dict, handle, protocol = -1)
        else:
            log.error("The dataset you try to save does not contain any data.")
            quit()


    def load(self, path):

        if not path: 
            log.error('Path to dataset pickle file to load is empty.')
            quit()

        if os.path.exists(path):
            log.info(f"Loading pickled dataset from file '{path}'..")
            with open(path, 'rb') as handle:
                load_dict = pickle.load(handle)
        else:
            log.info(f'File {path} doesn\'t exists, exiting.')
            quit()


        # load sub dataset into dataset object
        
        if not args.fix_dataset:
            args.fix_dataset = ('train', 'test', 'val')
        else:
            log.info(f"Fixing pickled dataset(s) '{args.fix_dataset}'.")

        for subset in ['train', 'test', 'val']:

            if subset in load_dict:
                if args.fix_dataset:
                    if subset in args.fix_dataset:
                        setattr(self, subset, [ Data().from_dict(graph_dict) for graph_dict in load_dict[subset] ])                
                else:
                    setattr(self, subset, [ Data().from_dict(graph_dict) for graph_dict in load_dict[subset] ])
            else:
                log.info(f"Did not find any {subset} data in the dataset that is being loaded.")
                setattr(self, subset, None)
        
        # load additional info into dataset object
        self.class_balance = load_dict['class_balance']
        self.num_genes = load_dict['num_genes']
        self.base_labels = load_dict['base_labels']
        self.base_labels_raw = load_dict['base_labels_raw']
        self.sim_score_dict = load_dict['similarity_dict']


        if not self.test and not self.train:
            log.error("Failed to load dataset from file, check that file contains at least one of training or test data.")
            quit()
    
    
    def get(self, idx):
        return self.data_lst[idx]
