from src.preprocessing import load_gff, load_similarity_score, load_ribap_groups, build_edge_index, map_edge_weights, map_labels_to_edge_index, construct_neighbour_lst, generate_neighbour_edge_features, build_adjacency_vectors
from src.setup import log, args
from src.helper import separate_components, concat_graph_data
import torch, os, pickle
from torch_geometric.data import Dataset, Data
from rich.progress import track, Console, Progress
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils.convert import to_scipy_sparse_matrix

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

    def generate_graph_data(self):
        """Generate datastructures that represent the input graph from the input data."""

        genome_annotation_df_lst = []
        genome_name_lst = []
        num_genes = 0
        self.data_lst = []
        self.gene_id_position_dict = {}

        # load annotations from gff files and format to pandas dataframe
        for gff_file in track(self.gff_files, description='Loading annotation files..', transient=True):
            genome_annotation_df = load_gff(gff_file)
            genome_annotation_df_lst.append(genome_annotation_df)
            log.info(f"Loaded annotation file of genome number {len(genome_annotation_df_lst)}: {gff_file}")
            log.debug(f"Genome 1 annotation dataframe:\n {genome_annotation_df}")
            num_genes += len(genome_annotation_df.index)
            
            self.gene_str_ids_lst += list(genome_annotation_df.index)
            
            # for each string gene ID save its normalized position in the gff file into the dictionary
            self.gene_id_position_dict.update({gene: (idx / len(list(genome_annotation_df.index))) for idx, gene in enumerate(list(genome_annotation_df.index))})


            genome_name_lst.append(os.path.basename(gff_file).split('.')[0].replace('_RENAMED', ''))

        self.neighbour_lst = construct_neighbour_lst(num_genes, self.num_neighbours)

        # total number of genes found in all annotation files
        log.info(f"Total number of genes found in annotation files: {num_genes}")
        self.gene_id_integer_dict = {gene: idx for idx, gene in enumerate(self.gene_str_ids_lst)}
        self.gene_ids_ts = torch.tensor(list(self.gene_id_integer_dict.values()))
        normalized_gene_positions_ts = torch.tensor([pos * num_genes for pos in list(self.gene_id_position_dict.values())]).unsqueeze(1)

        # load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
        sim_score_dict = load_similarity_score(self.similarity_score_file)

        with Console().status("Building edge index..") as status:
            self.edge_index_ts = build_edge_index(sim_score_dict, self.gene_id_integer_dict, fully_connected = False)
            log.info('Successfully built edge index')
        
        with Console().status("Mapping edge weights to respective edge index positions..") as status:
            self.edge_weight_ts = map_edge_weights(self.edge_index_ts, sim_score_dict, self.gene_str_ids_lst) #torch.randn((num_genes/2, edge_feature_dim))  # Edge 
            log.info('Successfully mapped weights to the edge index')

        #self.neighbour_edge_weights_ts = generate_neighbour_edge_features(self.neighbour_lst, self.edge_index_ts, sim_score_dict, self.gene_str_ids_lst)
        self.neighbour_edge_weights_ts = None
        
        #self.adjacency_vectors_ts = build_adjacency_vectors(num_neighbours = 2, gene_id_lst = self.gene_ids_ts)


        if self.ribap_groups_file:

            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict = load_ribap_groups(self.ribap_groups_file, genome_name_lst)

            # construct list of labels from ribap groups and format to match edge_index
            with Console().status("Mapping labels to gene pairs in edge index.") as status:
                self.labels_ts = map_labels_to_edge_index(self.edge_index_ts, self.gene_str_ids_lst, self.ribap_groups_dict)
                log.info(f"{self.labels_ts.sum().item() / len(self.labels_ts) * 100} % of labels are in positive class.")
                log.info('Successfully mapped labels to gene pairs in edge index')
        else:
            self.labels_ts = None

        if args.batch_size == 1:
            log.info('Batch size set to 1, skipping seperation of connected components.')
            graph_data = Data(normalized_gene_positions_ts, self.edge_index_ts, self.edge_weight_ts, self.labels_ts)
            #graph_data = Data(self.gene_ids_ts, self.edge_index_ts, self.edge_weight_ts, self.labels_ts)
            self.data_lst.append(graph_data)
            return
        
        #connected_components = separate_components(self.edge_index_ts)

        adj_matrix = to_scipy_sparse_matrix(self.edge_index_ts)
        graph = csr_array(adj_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        log.debug(f"Number of connected components: {n_components}, in total number of nodes: {num_genes}")

        connected_components_nodes = [[] for x in range(n_components)]
        
        for idx, label in enumerate(labels):
            connected_components_nodes[label].append(idx)
    
        with Progress(transient = True) as progress:
            subgraph_bar = progress.add_task("Generating subgraphs from connected components..", total=len(connected_components_nodes))


            for idx, component_nodes in enumerate(connected_components_nodes):
                x = torch.tensor(component_nodes).unsqueeze(1)
                # x is a tensor of categorical node IDs where a nodes index is in edge_index if it is connected to an edge,
                # so we have to either remap the indices in edge_index to the new x, or use the whole tensor for each data 
                # object so the indices work out 

                log.debug(f"subsetting similarity score dict for component nodes.. {component_nodes} {idx/len(connected_components_nodes) *100} % done.")

                gene_str_ids_lst = [self.gene_str_ids_lst[i] for i in component_nodes]
                component_normalized_gene_positions_ts = torch.tensor([normalized_gene_positions_ts[i] for i in component_nodes])

                sub_sim_score_dict = dict((gene_str_id, sim_score_dict[gene_str_id]) for gene_str_id in gene_str_ids_lst if gene_str_id in sim_score_dict)

                component_edge_index = build_edge_index(sub_sim_score_dict, self.gene_id_integer_dict, fully_connected = False)
                component_edge_weight_ts = map_edge_weights(component_edge_index, sub_sim_score_dict, self.gene_str_ids_lst)

                if self.ribap_groups_file:
                    component_labels_ts = map_labels_to_edge_index(component_edge_index, self.gene_str_ids_lst, self.ribap_groups_dict)
                else:
                    component_labels_ts = None

                # we use as node list for each graph the whole list since the edge index referes to indices in the node list, otherwise we have to remap all edge indices to their
                # match the updated sub list of nodes of the graph, which I would like to avoid (I hope this doesnt affect the model since it should anyway only make predictions for nodes that are connected by an edge?)
                # EDIT: as you can see i added tghe mapping to the sub graph nodes..
                node_sub_graph_mapping = None
                if len(component_edge_index[0]) > 0:

                    node_sub_graph_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(component_nodes)}

                    for idx in range(len(component_edge_index[0])):
                        edge = [component_edge_index[0][idx], component_edge_index[1][idx]]
                        component_edge_index[0][idx] = node_sub_graph_mapping[edge[0].item()]
                        component_edge_index[1][idx] = node_sub_graph_mapping[edge[1].item()]
                
                #graph_data = Data(self.gene_ids_ts.unsqueeze(1), component_edge_index, component_edge_weight_ts, component_labels_ts)
                graph_data = Data(component_normalized_gene_positions_ts.unsqueeze(1), component_edge_index, component_edge_weight_ts, component_labels_ts)
                graph.neighbour_edge_weights_ts = None # neighbour_edge_weights_ts
                self.data_lst.append(graph_data)
                progress.update(subgraph_bar, advance=1)

            log.info('Successfully generated graph data for all sub-graphs in the input')
        
        
        #data = Data(normalized_gene_positions_ts, self.edge_index_ts, self.edge_weight_ts, self.labels_ts)
        #data.neighbour_edge_weights_ts = self.neighbour_edge_weights_ts
        #self.data_lst = [data]
        

        #self.x = normalized_gene_positions_ts
        #self.x = self.adjacency_vectors_ts
        #self.x = self.gene_ids_ts
        #self.edge_attr = self.edge_weight_ts
        #self.edge_index = self.edge_index_ts
        #self.y = self.labels_ts

    
    def split_data(self, split = (0.7, 0.15, 0.15), batch_size = 32):
        """Split the singular graph data into train, test and validations sets, also create batches in the train dataset.

        Args:
            split (tuple, optional): . fraction of the graphs to go to train, test and validation sets respectively. Defaults to (0.7, 0.15, 0.15).
            batch_size (int, optional): number of graphs to trasin on in a single batch. Defaults to 32.
        """
        if args.batch_size == 1:
            log.info('Batch size set to 1, train and test datasets are the same.')
            self.train = self.data_lst
            self.test = self.data_lst[0]
            return
        
        # calculate train, test, val split and batches for train data
        num_train_data = int(len(self.data_lst) * split[0])
        num_test_data = int(len(self.data_lst) * split[1])
        log.info(f"Splitting datasets into sets of length {num_train_data}, {num_test_data}, {len(self.data_lst)-(num_test_data+num_train_data)}")
        self.train = self.data_lst[:num_train_data]
        self.train = [concat_graph_data(self.train[i:i + batch_size]) for i in range(0, len(self.train), batch_size)]
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
