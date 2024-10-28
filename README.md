# panGNN
Othologous gene prediction for pan genome construction using graph neural networks.


# how it works
Don't know but the idea is to use not only the commonly used sequence identity to predict whether two genes are orthologs (since this can be a bad indicator for species with high genetic variance or varying species strains) but also include the neighbourhood information of both genes. We assume here ortholog genes that did not originate from duplication events or transposition in the genome will share a similar genetic neighbourhood (the neighbour genes will be orthologs as well or have a certain sequence identity).

We gene annotation from Prokka for the gene neighbourhood information and run MMSeq on this annotation to cluster genes with similar sequences. This somehow has to be incorporated into a datastructure that we can feed to the graph neural network. Then do some smart convolution and design a training and test dataset. Profit
