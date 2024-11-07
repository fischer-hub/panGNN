# panGNN
Othologous gene prediction for pan genome construction using graph neural networks.


# how it works
The idea is to use not only the commonly used sequence identity to predict whether two genes are orthologs (since this can be a bad indicator for species with high genetic variance or varying species strains) but also include the neighbourhood information of both genes. We assume here ortholog genes that did not originate from duplication events or transposition in the genome will share a similar genetic neighbourhood (the neighbour genes will be orthologs as well or have a certain sequence identity).

We gene annotation from Prokka for the gene neighbourhood information and run MMSeq on this annotation to cluster genes with similar sequences. This somehow has to be incorporated into a datastructure that we can feed to the graph neural network. Then do some smart convolution and design a training and test dataset. Profit

# current state

The current design is shown below, where we have as input two input genomes a and b. For each gene in these genomes a node is created holding its two (for now) neighbour genes as described above, where the Prokka gene IDs of the actual gene in that node is mapped to an integer ID and embedded in a feature vector. Additionally the two neighbour gene IDs are also integer encoded and the embedding vectors are concatenated to hold the information of both the actual gene and its neighbours. If no neirghbour is available a 0 vector is used for the embedding. The network is fully connected (there should also be self looping edges but I didnt know how to properly draw thos in draw.io, Id like to test whether these self loops anyway give any information for prediction accuracy). The edges of each node pair hold the similarity bit score caluclated by MMSeqs2 (or 0 if no score was calculated, although this could bias the prediction depending on which threshold MMSeqs2 uses to filter low similarity pairs I think?).

Would then also be interesting if the design works for pairwise species homology predictions if we can scale it to take more genomes at once, e.g.: the bit scores we already have for all input genomes at once anyway. Not sure if it works so easy to input such a multitude bigger feature space. But for now lets get it running on this minimal example.


![Alt text](./assets/panGNN_example_graph.svg)

# get started
Install the dependencies first using e.g. the conda environment file in the project directory and activate conda environment:

```
mamba env  create -f pangnn.yaml
conda activate panGNN
```

Then run pangnn.py:
```
python pangnn.py
```

But dont expect too much, currently there is only some preprocessing happening on the test data in `/data` (although very slow).