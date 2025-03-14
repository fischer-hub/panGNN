# panGNN
Othologous gene prediction for pangenome construction using graph neural networks.


# how it works
The idea is to use not only the commonly used sequence identity to predict whether two genes are orthologs (since this can be a bad indicator for species with high genetic variance or varying species strains) but also include the neighbourhood information of both genes (_gene synteny_). We assume that ortholog genes that did not originate from duplication events or transposition in the genome will share a similar genetic neighbourhood (the neighbour genes will be orthologs as well or have a certain sequence identity).

To obtain gene annotations and neighbourhood information, we used Prokka and subsequently ran MMSeq2 on these annotations to cluster genes based on sequence similarity. Next, we aim to transform this information into a data structure that we can feed to the graph neural network (GNN). Then, we plan to do some smart convolution and design a training and test dataset. Profit.

Now, does this information separate the data points somehow?

![Alt text](./plots/scatter_sim_score_dist.png)

Not linearly..

![Alt text](./plots/umap_sim_score_dist.png)
![Alt text](./plots/umap3d_sim_score_dist.png)

Maybe non linearly?



# current state

Currently, the genes are loaded from the GFF files, the similarity scores are loaded from the MMSeqs2 output and used to generate edges between all pairs of genes with a similarity score. The RIBAP output table (holy table) is then used to check for which pairs of genes both genes are part of the same RIBAP group; these are assumed to be of prediction class 1 (homolog). The similarity scores are mapped to the edge weights of the input graph in the PyG Data object. Additionally, the normalized position of each gene (the position in its GFF file normalized by the total number of genes in that file) is used as node features. Then, every gene is represented by a node containing its normalized position and is connected by edges that represent the similarity bit score to its connected gene (node). The idea here is, that genes that are in close proximity in their genomes will be learned to have a higher chance to be homolog when their similarity score is also high, compared to high similarity scores but very different positional node features.

The similarity input graph is then separated into all individual connected components (sub-graphs), see:

![Alt text](./plots/input_graph.png)

Since genes below the MMSeqs2 similarity threshold will not have an edge connecting them in the input graph, they will not have any influence on each other and the training weights (at least, I hope so) in the GNN. The convolution can only occur between nodes connected by a node, and only nodes with a similarity score will be used in the encoding and prediction step. So we can use this to separate the data into small bits and batch them in groups that we want to train on (also, we can use some part of the data like that for test and validation set for now) without destroying relationships in the data (e.g., by randomly subsampling graphs and breaking edges). The order of the batches is also shuffled during the model training, reducing the risk of fitting the model to the order of the data instead of the data itself.

The first layer of the GNN is a linear (embedding) layer, putting the one-dimensional node (input) features into the hidden dim space (which is 64 for now). This is followed by a convolution layer that incorporates the edge weights of each node. It performs a weighted sum of the edge weights connected to each node before being multiplied with the learnable parameter matrix. Next is a ReLU activation layer and another convolution layer. After this, the node embeddings are decoded by calculating the dot product between pairs of node embeddings to predict links between them. The resulting values (logits, if you will so in ML slang, I guess) are then fed to a binary cross entropy loss function that applies a sigmoid function to the logits before calculating the loss between them and the 'ground truth' (binary) labels.

However, it is not performing very well right now.

# get started
Install the dependencies first using e.g. the conda environment file in the project directory and activate conda environment:

```bash
mamba env create -f pangnn.yaml
conda activate panGNN
```

Then run pangnn.py:

```bash
python pangnn.py
```

But dont expect too much, currently there is only some preprocessing happening on the test data in `/data` (although very slow).