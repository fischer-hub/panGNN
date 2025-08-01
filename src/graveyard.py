### some boilerplate code for GNN design which does not use pytorch geometric though
# Define the GNN model
class GeneHomologyGNN(torch.nn.Module):
    def __init__(self, num_genes, embedding_dim, edge_feature_dim, hidden_dim, output_dim=1):
        super(GeneHomologyGNN, self).__init__()

        # Embedding layer to convert gene IDs to vectors
        self.embedding = torch.nn.Embedding(num_genes, embedding_dim)
        
        # Graph convolutional layers
        self.conv1 = GCNConv(3 * embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for classification based on the graph-level output
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, gene_ids, edge_index, edge_attr, batch, neighbor_dict):
        # Convert gene IDs to embeddings
        gene_embeddings = self.embedding(gene_ids)
        
        # Prepare node features by combining with neighbor embeddings
        node_features = self.prepare_node_features_with_neighbors(gene_embeddings, neighbor_dict)
        
        # Pass through the graph convolution layers
        x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pooling the features across the graph
        x = global_mean_pool(x, batch)
        
        # Classification layer
        out = self.classifier(x)
        return out

    def prepare_node_features_with_neighbors(self, gene_embeddings, neighbor_lst):
        """
        Combine each gene's embedding with its neighbors' embeddings.
        
        Parameters:
        - gene_embeddings: Tensor of shape [num_genes, embedding_dim]
        - neighbor_lst: List where indices are gene integer indices and elems are tuples
          of the form (-i-th-upstream_neighbor,... , ith-downstream_neighbor).
          
        Returns:
        - node_features: Tensor with combined neighbor features for each gene.
        """
        num_genes, embedding_dim = gene_embeddings.shape
        combined_features = []

        for gene_id in range(num_genes):
            # Get the current gene's embedding
            gene_feature = gene_embeddings[gene_id]
            
            # TODO: adjust this to work with more than 1 neighbour
            # get neighbors' embeddings
            upstream_id, downstream_id = neighbor_lst[gene_id]
            upstream_feature= gene_embeddings[upstream_id] if upstream_id is not None else torch.zeros(embedding_dim)
            downstream_feature= gene_embeddings[downstream_id] if downstream_id is not None else torch.zeros(embedding_dim)
            
            # Concatenate embeddings (upstream gene + gene + downstream gene)
            combined_feature= torch.cat([upstream_feature, gene_feature, downstream_feature], dim=0)
            combined_features.append(combined_feature)

        return torch.stack(combined_features)

# first class to hold graph dataset 
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