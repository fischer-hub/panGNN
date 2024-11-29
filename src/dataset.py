from src.preprocessing import load_gff, load_similarity_score, load_ribap_groups, build_edge_index, map_edge_weights, map_labels_to_edge_index, construct_neighbour_lst, generate_neighbour_edge_features
from src.setup import log
import torch, os
from torch_geometric.data import Dataset, Data

class HomogenousDataset(Dataset):
    def __init__(self, gff_files = [], similarity_score_file = '', ribap_groups_file = None, num_neighbours = 1):
        super().__init__(root = None, transform = None, pre_transform = None, pre_filter = None)

        self.gene_ids_lst = None
        self.gene_id_integer_dict = None
        self.gene_ids_ts = None

        # load annotations from gff files and format to pandas dataframe
        genome1_name = gff_files[0]
        genome1_annotation_df = load_gff(genome1_name)
        log.info(f"Loaded annotation file of first genome: {genome1_name}")
        log.debug(f"Genome 1 annotation dataframe:\n {genome1_annotation_df}")

        genome2_name = gff_files[1]
        genome2_annotation_df = load_gff(genome2_name)
        log.info(f"Loaded annotation file of second genome: {genome2_name}")
        log.debug(f"Genome 2 annotation dataframe:\n {genome2_annotation_df}")

        # total number of genes found in all annotation files
        num_genes = len(genome1_annotation_df.index) + len(genome2_annotation_df.index)
        log.info(f"Total number of genes found in annotation files: {num_genes}")

        self.gene_ids_lst = list(genome1_annotation_df.index) + list(genome2_annotation_df.index)
        self.gene_id_integer_dict = {gene: idx for idx, gene in enumerate(self.gene_ids_lst)}
        self.gene_ids_ts = torch.tensor(list(self.gene_id_integer_dict.values()))

        # load similarity bit scores from MMSeqs2 output CSV file to pandas dataframe
        sim_score_dict = load_similarity_score(similarity_score_file)
        log.info(f"Loaded similarity scores file: {similarity_score_file}")

        self.edge_index_ts = build_edge_index(sim_score_dict, self.gene_id_integer_dict, fully_connected = False)
        log.info(f"Edge index for fully connected graph successfully created.")
        
        self.edge_weight_ts = map_edge_weights(self.edge_index_ts, sim_score_dict, self.gene_ids_lst)#torch.randn((num_genes/2, edge_feature_dim))  # Edge 
        
        self.neighbour_lst = construct_neighbour_lst(len(genome1_annotation_df.index)) + construct_neighbour_lst(len(genome2_annotation_df.index), num_neighbours = num_neighbours)

        self.neighbour_edge_weights = generate_neighbour_edge_features(self.neighbour_lst, self.edge_index_ts)


        if ribap_groups_file:
            # load holy ribap table to generate labels for test data set
            self.ribap_groups_dict = load_ribap_groups(ribap_groups_file, [os.path.basename(genome1_name).split('.')[0].replace('_RENAMED', ''), os.path.basename(genome2_name).split('.')[0].replace('_RENAMED', '')])
            log.info(f"Loaded RIBAP groups file: {ribap_groups_file}")
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
        self.edge_weight_ts = self.edge_weight_ts * self.neighbour_edge_weights
    
    #def split_dataset(self, train, test, validation = 0):
    

