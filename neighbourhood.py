#!/usr/bin/env python3
from BCBio import GFF
import pandas as pd
import argparse, Bio, os

# argparse stuff
parser = argparse.ArgumentParser(
                    prog='neighbourhood.py',
                    description='Takes in GFF annotation file (e.g. from Prokka) and calculates matrix of neighbourhood (adjacency) matrix for each feature in GFF.',
                    epilog='Text at the bottom of help')

parser.add_argument('annotation_file_name', nargs = '?', default = os.path.join('data', 'Chlamydia_abortus_S26_3_strain_S26_3_full_genome_RENAMED.gff'))           # positional argument
parser.add_argument('-d', '--distance', help = 'Number of genes from target gene to consider as neighbours.', default = 1)
parser.add_argument('-k', '--basepair_threshold', help = 'Number of basepairs that two genes can be apart while still counting as neighbours.', default = 1000)
#parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
        
args = parser.parse_args()


# open GFF file (open test data if nothing provided on cmd arg)
# in_handle = open(args.annotation_file_name)
with open(args.annotation_file_name) as gff_handle:

#    gff = gff_handle.read()
#    
#    try:
#        gff = gff.split('##FASTA')[0]
#    except Exception as e:
#        print(f"Could not split GFF file at beginning of FASTA sequence: {e}\nMaybe the file does not contain a FASTA sequence after all.")

    annotation_df = pd.read_csv(gff_handle, comment = '#', sep = '\t', names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])

annotation_df = annotation_df.dropna()
annotation_df['gene_id'] = annotation_df.attribute.str.replace(';.*', '', regex = True)
annotation_df['gene_id'] = annotation_df.gene_id.str.replace('ID=', '', regex = True)
annotation_df.set_index('gene_id', inplace = True)



print(annotation_df[['start', 'end']])