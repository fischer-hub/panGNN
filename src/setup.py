from src.header import print_header
import argparse, logging, os
from rich.traceback import install
from rich.logging import RichHandler


# argparse stuff
parser = argparse.ArgumentParser(
                    prog='pangnn.py',
                    description='The heart and soul of PanGNN. TODO: write sometyhing useful here.',
                    epilog='Greta Garbo and Monroe, Dietrich and DiMaggio, Marlon Brando, Jimmy Dean, On the cover of a magazine.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general args
parser.add_argument('-d', '--debug',      help = 'set log level to DEBUG and print debug information while running', action='store_true')  # on/off flag
parser.add_argument('-p', '--plot_graph', help = "plot input and output graph and save in './plots'", action='store_true')  # on/off flag
parser.add_argument('-t', '--traceback',  help = 'turns on rich formatting of traceback', action = 'store_true')
parser.add_argument('-c', '--cache',  	  help = 'cache computionally slow data structures', action = 'store_true')
parser.add_argument('-l', '--log_level',  help = "set the level to print logs ['NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']", default = 'INFO', type = str)
parser.add_argument('-m', '--model_args', help = 'path to save or load model from, depending on training or prediction mode', default = 'model.pkl', type = str)
parser.add_argument('-n', '--neighbours', help = 'number of genes from target gene to consider as neighbours', default = 1, type = int)
parser.add_argument('-a', '--annotation', help = 'path to the two annotation files in gff format of the two input genomes, seperated by tab', default = [os.path.join("data", "Cga_08-1274-3_RENAMED.gff"), os.path.join("data", "Cga_12-4358_RENAMED.gff")], type = str, nargs = '*')
parser.add_argument('-s', '--similarity', help = 'path to the similarity score file (e.g tab seperated output of MMSeqs2)', default = os.path.join('data', 'mmseq2_result.csv'), type = str)
parser.add_argument('--binary_threshold', help = 'binary threshold to classify output probabilities to the label class', default = 0.5, type = float)
parser.add_argument('--dynamic_binary_threshold', help = 'dynamically calculate the binary threshold that separates the predictions best based on yuden index', action = 'store_true')
parser.add_argument('--simulate_dataset', help = 'parameters to generate simulated input data with seperated by whitespace [num_genes_per_genome, num_genomes, fraction_pos_edges, num_fragments, num_frags_to_shuffle]', nargs=5, type=str, default = None)
parser.add_argument('--simulated_score_means', help = 'means of the gamma distributions to draw the scores of negative and positive edge scores from during simulation seperated by whitespace [negative_mean, positive_mean]', nargs=2, type=int, default = [200, 500])
parser.add_argument('--union_edge_weights', help = 'unite edge weights from sim and neighbour graph and only convolute over one', action = 'store_true')
parser.add_argument('--include_trivial', help = 'include scores from trivial cases where there is only one candidate to predict the true homolog from', action = 'store_true')
parser.add_argument('--skip_connections', help = 'activate skip connections in model', action = 'store_true')
parser.add_argument('--categorical_node', help = 'embed node features as categorical feature embeddings using embedding layer, where category referes to the position of each gene in its genome', action = 'store_true')
parser.add_argument('--no_q_score_transform', help = 'dont transform normalized edge probabilities between homolog candidates to Q-score like values before training [default: True]', action = 'store_false')
parser.add_argument('--normalization_temp', help = 'temperature value for similarity score normalization, turns off normalization if set to 0', default = 0.8, type = float)
parser.add_argument('--tb_comment',         help = 'comment to append to current run for evaluation with tensorboard', default = '')
parser.add_argument('--from_pickle',        help = 'path to pickle file to load saved dataset from', default = '')
parser.add_argument('--to_pickle',          help = 'path to pickle file to save current dataset to', default = '')
parser.add_argument('--fix_dataset',        help = "for loaded dataset fix subset instead of generating it newly, chose from ['train', 'val', 'test']", default = [], type = str, nargs = '*')
parser.add_argument('--node_dim',           help = 'dimension of node embedding and input of first convolution layer', default = 64, type = int)
parser.add_argument('--hidden_dim',         help = 'dimension of hidden convoluytion layer(s)', default = 128, type = int)
parser.add_argument('--decoder',            help = "decoding strategy (similarity measure) to use predict link between two node embeddings ['mlp', 'cosine', 'dotproduct']", default = 'mlp', type = str)
parser.add_argument('--base_model',         help = "train using the base model that only convolutes over similarity edges", action='store_true')
parser.add_argument('-o', '--output',        help = "output directory to store run files (PR-AUC, tensorboard track files, etc.) in ", default = 'runs', type = str)

# train mode args
parser.add_argument('--train',              help = 'set pangnn into training mode', action='store_true')
parser.add_argument('-b', '--batch_size',   help = 'set number of graphes to be contained in one batch', default = 32, type = int)
parser.add_argument('-e', '--epochs',       help = 'set number of epochs for model training', default = 10, type = int)
parser.add_argument('-r', '--ribap_groups', help = 'path to file holding the ribap groups calculated for the input genomes', default = os.path.join('data', 'holy_python_ribap_95.csv'), type = str)
parser.add_argument('-@', '--cpus',         help = 'max number of threads used during preprocessing', default = 2, type = int)
parser.add_argument('--mixed_precision',    help = "mixed precision setting to use ['no', 'fp16', 'bf16']", default = 'no', type = str)

# parse args
args = parser.parse_args()
if args.simulate_dataset is not None:
    if len(args.simulate_dataset) < 5: log.error(f'Argument --simulate_dataset provided but only {len(args.simulate_dataset)} of 5 values were supplied. Exiting..')
    args.simulate_dataset = [int(args.simulate_dataset[0]), int(args.simulate_dataset[1]), float(args.simulate_dataset[2]), float(args.simulate_dataset[3]), float(args.simulate_dataset[4])]



# setup logger
FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level if not args.debug else 'DEBUG', format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")

logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logfile_handler = logging.FileHandler("pangnn.log", mode="w")
logfile_handler.setLevel(args.log_level)

log.addHandler(logfile_handler)

# basic sanity checks
if args.traceback: install(show_locals=True)

if args.train and not args.ribap_groups:
    log.error("Training mode was selected but no label data (RIBAP groups file) was supplied, cannot train model without 'ground truth' data, exiting.")
    quit()

if len(args.fix_dataset) > 3:
    log.error(f"More than 3 data sub sets defined ('{args.fix_dataset}') to be fixed, pleas chose from ['train', 'val', 'test'] only.")
    quit()

for subset in args.fix_dataset:
    if subset not in ['train', 'val', 'test']:
        log.error(f"Subset '{subset}' is not a valid subset name, please chose from ['train', 'val', 'test'].")
        quit()


# print header text
print_header(True, args)

hparams = {
    "batch_size": args.batch_size,
    "num_epochs": args.epochs,
    "num_neighbours": args.neighbours,
    "binary_threshold": args.binary_threshold,
    "dynamic_binary_threshold": args.dynamic_binary_threshold,  
    "simulate_dataset": args.simulate_dataset,
    "categorical_node": args.categorical_node,
    'no_q_score_transform': args.no_q_score_transform,
    'normalization_temp': args.normalization_temp,
    'tb_comment': args.tb_comment
}
