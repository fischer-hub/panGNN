from src.preprocessing import load_gff, load_similarity_score, load_ribap_groups, build_edge_index, map_edge_weights, map_labels_to_edge_index, construct_neighbour_lst, generate_neighbour_edge_features
from src.setup import log
import torch, os
from torch_geometric.data import Dataset, Data
from rich.progress import track


class HomogenousDataset(Dataset):
    def __init__(self, gff_files = [], similarity_score_file = '', ribap_groups_file = None, num_neighbours = 1):
        super().__init__(root = None, transform = None, pre_transform = None, pre_filter = None)

        self.gff_files = gff_files
        self.similarity_score_file = similarity_score_file
        self.num_neighbours = num_neighbours
        self.ribap_groups_file = ribap_groups_file
        self.data_lst = []
        self.gene_ids_lst = []
        self.neighbour_lst = []


    def generate_graph_data(self):
        """Generate datastructurres that represent the input graph from the input data."""

        genome_annotation_df_lst = []
        genome_name_lst = []
        num_genes = 0

        # load annotations from gff files and format to pandas dataframe
        for gff_file in track(self.gff_files, description='Loading annotation files..', transient=True):
            genome_annotation_df = load_gff(gff_file)
            genome_annotation_df_lst.append(genome_annotation_df)
            log.info(f"Loaded annotation file of first genome: {gff_file}")
            log.debug(f"Genome 1 annotation dataframe:\n {genome_annotation_df}")
            num_genes += len(genome_annotation_df.index)
            
            self.gene_ids_lst += list(genome_annotation_df.index)

            genome_name_lst.append(os.path.basename(gff_file).split('.')[0].replace('_RENAMED', ''))

        
        self.neighbour_lst = construct_neighbour_lst(num_genes, self.num_neighbours)

        # total number of genes found in all annotation files
        log.info(f"Total number of genes found in annotation files: {num_genes}")
        self.gene_id_integer_dict = {gene: idx for idx, gene in enumerate(self.gene_ids_lst)}
        self.gene_ids_ts = torch.tensor(list(self.gene_id_integer_dict.values()))

        # load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
        sim_score_dict = load_similarity_score(self.similarity_score_file)
        log.info(f"Loaded similarity scores file: {self.similarity_score_file}")

        self.edge_index_ts = build_edge_index(sim_score_dict, self.gene_id_integer_dict, fully_connected = False)
        log.info(f"Edge index successfully created.")
        
        self.edge_weight_ts = map_edge_weights(self.edge_index_ts, sim_score_dict, self.gene_ids_lst) #torch.randn((num_genes/2, edge_feature_dim))  # Edge 
        self.neighbour_edge_weights_ts = generate_neighbour_edge_features(self.neighbour_lst, self.edge_index_ts, sim_score_dict, self.gene_ids_lst)


        if self.ribap_groups_file:
            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict = load_ribap_groups(self.ribap_groups_file, genome_name_lst)
            log.info(f"Loaded RIBAP groups file: {self.ribap_groups_file}")
            log.debug(f"Got RIBAP groups dictionary:\n {next(iter(self.ribap_groups_dict.items()))}")

            # construct list of labels from ribap groups and format to match edge_index
            self.labels_ts = map_labels_to_edge_index(self.edge_index_ts, self.gene_ids_lst, self.ribap_groups_dict)
            log.info('Created tensor of labels for training from RIBAP groups.')
        else:
            self.labels_ts = None

        self.x = self.gene_ids_ts
        self.edge_attr = self.edge_weight_ts
        self.edge_index = self.edge_index_ts
        self.y = self.labels_ts
    
    def len(self):
        return len(self.gene_ids_lst)

    def get(self, idx):
        return Data(self.gene_ids_ts, self.edge_index_ts, self.edge_weight_ts, self.labels_ts)
    
    def to(self, device):
        self.x.to(device)
        self.edge_attr.to(device)
        self.edge_index.to(device)
        self.y.to(device)

    def scale_weights(self):
        """Scale the similarity scores on the edges of the input graph by the gene neighbourhood similarity factor
        """
        self.edge_weight_ts = self.edge_weight_ts * self.neighbour_edge_weights_ts
        self.edge_attr = self.edge_weight_ts

    
    #def split_dataset(self, train, test, validation = 0):
    

