put the fries in the bag take the cheese out of the mousetrap

# About panGNN
As part of my masters thesis on ortholog gene prediction with graph neural networks, `panGNN` is the core piece of software implementing the developed model and datastructure as well as taking care of both the training of new models and inference of those models on new input data. The idea is to use ILP refined ortholog gene clusters from the output of the [`RIBAP`](https://github.com/hoelzer-lab/ribap) pipeline (that is limited by the input size because of ILP solving complexity) to train a graph neural network that will be able to approximate the refined ILP solution on bigger datasets in reasonable time.

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