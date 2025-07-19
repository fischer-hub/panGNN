put the fries in the bag take the cheese out of the mousetrap

# About panGNN
As part of my masters thesis on ortholog gene prediction with graph neural networks, `panGNN` is the core piece of software implementing the developed model and datastructure as well as taking care of both the training of new models and inference of those models on new input data. Therefore sequence similarity scores and synteny of the input genes are used to infer orthology in the input data. The idea is to use ILP refined ortholog gene clusters from the output of the [`RIBAP`](https://github.com/hoelzer-lab/ribap) pipeline (that is limited by the input size because of ILP solving complexity) to train a graph neural network that will be able to approximate the refined ILP solution on bigger datasets in reasonable time.

# Getting started
## Insatallation
To run `panGNN` you have to install the dependencies listen in `pangnn.yaml`. You can do this manually or using `conda/mamba` as follows:

```
conda env create -f pangnn.yaml
```
NOTE: If you need GPU support e.g. for training a new model or big input datasets you might have to install `accelerate` and / or `PyTorch` manually before creating the conda environment. This is a [common issue](https://discuss.huggingface.co/t/accelerate-doesnt-seem-to-use-my-gpu/79952) on some computing setups (especially on HPCs and similar shared computing environments). Once you are set up `panGNN` will prefer GPU usage over CPU usage and will indicate the device used during startup with:
```
INFO     Training on device: cuda
```
which means your GPU was detected successfully or with
```
INFO     Training on device: cpu:0
```
indicating the CPU is used.

Finally you have to activate the freshly installed conda environment:
```
conda activate panGNN
```

## Run panGNN
Check if your installation is working by running a minimal command:
```
accelerate launch pangnn.py --train
```
This will by default produce a `model.pkl` holding the trained model parameters that can be found in the `runs` directory in the respective sub directory with the latest timestamp as the directory name.

You can then infer the trained model with:
```
accelerate launch pangnn.py -m path/to/your/model.pkl

```

# Options
There is quite a lot of options to change, especially if you want to train a new model:

```
  -h, --help            show this help message and exit
  -d, --debug           set log level to DEBUG and print debug information while running (default: False)
  -p, --plot_graph      plot input and output graph and save in './plots' (default: False)
  -t, --traceback       set traceback standard python format (turns on rich formatting of traceback) (default: False)
  -c, --cache           cache computionally slow data structures (default: False)
  -l LOG_LEVEL, --log_level LOG_LEVEL
                        set the level to print logs ['NOTSET', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'] (default: INFO)
  -m MODEL_ARGS, --model_args MODEL_ARGS
                        path to save or load model from, depending on training or prediction mode (default: model.pkl)
  -n NEIGHBOURS, --neighbours NEIGHBOURS
                        number of genes from target gene to consider as neighbours (default: 1)
  -a [ANNOTATION ...], --annotation [ANNOTATION ...]
                        path to the two annotation files in gff format of the two input genomes, seperated by tab (default: ['data/Cga_08-1274-3_RENAMED.gff', 'data/Cga_12-4358_RENAMED.gff'])
  -s SIMILARITY, --similarity SIMILARITY
                        path to the similarity score file (e.g tab seperated output of MMSeqs2) (default: data/mmseq2_result.csv)
  --binary_threshold BINARY_THRESHOLD
                        binary threshold to classify output probabilities to the label class (default: 0.5)
  --dynamic_binary_threshold
                        dynamically calculate the binary threshold that separates the predictions best based on yuden index (default: False)
  --simulate_dataset SIMULATE_DATASET SIMULATE_DATASET SIMULATE_DATASET SIMULATE_DATASET SIMULATE_DATASET
                        parameters to generate simulated input data with seperated by whitespace [num_genes_per_genome, num_genomes, fraction_pos_edges, num_fragments, num_frags_to_shuffle] (default: None)
  --simulated_score_means SIMULATED_SCORE_MEANS SIMULATED_SCORE_MEANS
                        means of the gamma distributions to draw the scores of negative and positive edge scores from during simulation seperated by whitespace [negative_mean, positive_mean] (default: [200,
                        500])
  --union_edge_weights  unite edge weights from sim and neighbour graph and only convolute over one (default: False)
  --exclude_trivial     exclude scores from trivial cases where there is only one candidate to predict the true homolog from (default: False)
  --skip_connections    activate skip connections in model (default: False)
  --categorical_node    embed node features as categorical feature embeddings using embedding layer, where category referes to the position of each gene in its genome (default: False)
  --no_q_score_transform
                        dont transform normalized edge probabilities between homolog candidates to Q-score like values before training [default: True] (default: True)
  --normalization_temp NORMALIZATION_TEMP
                        temperature value for similarity score normalization, turns off normalization if set to 0 (default: 0.8)
  --tb_comment TB_COMMENT
                        comment to append to current run for evaluation with tensorboard (default: )
  --from_pickle FROM_PICKLE
                        path to pickle file to load saved dataset from (default: )
  --to_pickle TO_PICKLE
                        path to pickle file to save current dataset to (default: )
  --fix_dataset [FIX_DATASET ...]
                        for loaded dataset fix subset instead of generating it newly, chose from ['train', 'val', 'test'] (default: [])
  --node_dim NODE_DIM   dimension of node embedding and input of first convolution layer (default: 64)
  --hidden_dim HIDDEN_DIM
                        dimension of hidden convoluytion layer(s) (default: 128)
  --decoder DECODER     decoding strategy (similarity measure) to use predict link between two node embeddings ['mlp', 'cosine', 'dotproduct'] (default: mlp)
  --base_model          train using the base model that only convolutes over similarity edges (default: False)
  -o OUTPUT, --output OUTPUT
                        output directory to store run files (PR-AUC, tensorboard track files, etc.) in (default: runs)
  --train               set pangnn into training mode (default: False)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        set number of graphes to be contained in one batch (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        set number of epochs for model training (default: 10)
  -r RIBAP_GROUPS, --ribap_groups RIBAP_GROUPS
                        path to file holding the ribap groups calculated for the input genomes (default: data/holy_python_ribap_95.csv)
  -@ CPUS, --cpus CPUS  max number of threads used during preprocessing (default: 2)
  --mixed_precision MIXED_PRECISION
                        mixed precision setting to use ['no', 'fp16', 'bf16'] (default: no)
```