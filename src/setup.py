from src.header import print_header
import argparse, logging, os
from rich.traceback import install
from rich.logging import RichHandler


# print header text
print_header(True)

# argparse stuff
parser = argparse.ArgumentParser(
                    prog='pangnn.py',
                    description='The heart and soul of PanGNN. TODO: write sometyhing useful here.',
                    epilog='Greta Garbo and Monroe, Dietrich and DiMaggio, Marlon Brando, Jimmy Dean, On the cover of a magazine.',
                    usage='pangnn.py [ train | predict ] [ options ]',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(dest="command", help="Available commands")

parser.add_argument('-d', '--debug',      help = 'set log level to DEBUG and print debug information while running', action='store_true')  # on/off flag
parser.add_argument('-t', '--traceback',  help = 'set traceback standard python format (turns off rich formatting of traceback)', action = 'store_true')
parser.add_argument('-l', '--log_level',  help = "set the level to print logs ['NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']", default = 'INFO', type = str)
parser.add_argument('-m', '--model_args', help = 'path to save or load model from, depending on training or prediction mode', default = 'model.pkl', type = str)
parser.add_argument('-n', '--neighbours', help = 'number of genes from target gene to consider as neighbours', default = 1, type = int)
parser.add_argument('-a', '--annotation', help = 'path to the two annotation files in gff format of the two input genomes, seperated by comma', default = 'data/dummy_dataset/dummy1.gff	data/dummy_dataset/dummy2.gff', type = str)
parser.add_argument('-s', '--similarity', help = 'path to the similarity score file (e.g tab seperated output of MMSeqs2)', default = os.path.join('data', 'dummy_dataset', 'dummy_mmseqs2.csv'), type = str)


train_parser = subparsers.add_parser("train",     help="train a model on a input dataset")
train_parser.add_argument('-b', '--batch_size',   help = 'set dataset batch size for model training', default = 32, type = int)
train_parser.add_argument('-e', '--epochs',       help = 'set number of epochs for model training', default = 1, type = int)
train_parser.add_argument('-r', '--ribap_groups', help = 'path to file holding the ribap groups calculated for the input genomes', default = os.path.join('data', 'dummy_dataset', 'dummy_ribap.csv'), type = str)

prediction_parser = subparsers.add_parser("predict", help="infer a trained model and predict homolog genes in the input dataset")

args = parser.parse_args()

# setup logger
FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level if not args.debug else 'DEBUG', format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")
if not args.traceback: install(show_locals=True)