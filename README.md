# ICLR25 | Denoising Lévy Probabilistic Models (DLPM)

[Arxiv paper](https://arxiv.org/abs/2407.18609)

Denoising Levy Probabilistic Models are heavy-tailed generative diffusion models, replacing the Gaussian distribution by the [Lévy stable distribution](https://en.wikipedia.org/wiki/Stable_distribution). This heavy-tailed modification improves the generation of heavy-tailed datasets, or (bounded) datasets exhibiting strong class imbalance (e.g., images with rare classes).

This repository contains the full implementation of DLPM, which consists in small modifications on top of an existing Gaussian-based diffusion model (DDPM) implementation.

## Repository Overview

This repo provides the tools for training, evaluation, and generation of data using these models. We are using [BEM (Better Experimentation Manager for deep learning with Pytorch)](https://github.com/darioShar/bem) to manage our experiments.

- **Generative Model Implementation**: Located in `dlpm/methods/dlpm.py` and `dlpm/methods/GenerativeLevyProcess.py`, these files contain the core logic of the DLPM generative model. Users interested in understanding or modifying the underlying generative processes should start here.
  
- **Neural Network Architecture**: If you want to change the neural networks used in the DLPM, head to the `dlpm/models` directory. This is where all the neural network models are defined and can be customized according to your needs.
  
- **Logging Configuration**: To customize how logging is handled, you can heritate from `bem/Logger.py` for integrating your own logger. An example of a custom Weights & Biases logging setup is available in `dlpm/NeptuneLogger.py`.

- **Experiment Workflow**: The `dlpm/dlpm_experiment` file orchestrates the initialization of the training, evaluation, and data generation processes. To integrate your modifications into the experiment flow, update the `init` functions here. These functions will be provided to the `Experiment` class from the `bem` library.

- **Configuration Parameters**: Parameters for specific runs are passed in a dictionary called `p`, which is loaded from configuration files located in `dlpm/configs/`. Adjust these files to fine-tune the behavior of the model and experiment settings.

- **Comparison between discrete time DLPM and continuous time LIM**: This repository supports both DLPM, our discrete time heavy-tailed diffusion model, and LIM, the continuous time counterpart, as retrieved from the supplmentary material of the [corresponding LIM paper](https://openreview.net/forum?id=0Wp3VHX0Gm). While our paper argues for the superiority of the DLPM approach, users can experiment with both models and compare their behavior and performance.

## Supported Datasets

Here’s a brief overview of the supported datasets, as provided by `BEM`, and how to specify them:

- **2D Datasets**: 
  - The repository supports synthetic 2D datasets. See `Generator.available_distributions` in `bem/datasets/Data.py`.

- **Image Datasets**: 
  - You can use standard image datasets (MNIST, CIFAR-10, its long-tailed version CIFAR-10-LT, CELEBA etc.). See `bem/datasets/__init__.py`.

Configuration files for some of these datasets are provided in the `dlpm/configs/` directory: `mnist.yml` for MNIST, `cifar10.yml` for CIFAR-10, `cifar10_lt.yml` for CIFAR-10-LT.
    
You can modify the configuration files to adjust data loading settings, such as the batch size or data augmentation options, according to your experiment needs.

## Using the Provided Scripts

This repository includes scripts that simplify the process of training, evaluating, and visualizing the results of DLPM. Below is a description of each script and how to use them:

### 1. `run.py`

This script is used to train a model. It accepts various command-line arguments to control the training process, including configuration settings and experiment parameters.

**Example Command**:
```bash
python ./run.py --config mnist --name dlpm_test --method dlpm --epochs 100 --eval 50 --check 50 --train_reverse_steps 1000 
```

**Explanation**:
- `--config`: Specifies the configuration file to use (e.g., `mnist`).
- `--name`: The name of the experiment run, used for logging and identification. Here, the checkpointed models will be stored in `/models/dlpm_test/`.
- `--method`: Specifies the generative method to use (either `dlpm` or `lim`), in this case, `dlpm`.
- `--epochs`: The total number of training epochs.
- `--eval`: Specifies the interval (in epochs) for running evaluations during training.
- `--check`: Interval for model checkpointing (in epochs).
- `--train_reverse_steps`: The number of reverse steps to use during training.

### 2. `eval.py`

This script evaluates a pre-trained model and can also be used for generating samples from the trained model.

**Example Command**:
```bash
python ./eval.py --config mnist --name dlpm_test --method dlpm --epochs 100 --eval 100 --generate 2000 --reverse_steps 1000
```

**Explanation**:
- `--config`, `--name`, `--method`, and `--epochs`: Same as in `run.py`.
- `--eval`: Specifies the evaluation checkpoint to use.
- `--generate`: Number of samples to generate.
- `--reverse_steps`: Number of reverse steps to use during the generation process.

## Quick Reproduction Guide

The commands below assume you already cloned this repository and that you would like to work inside a Conda environment named `dlpm`.

### 1. Create & activate the Conda environment

```bash
conda create -n dlpm python=3.10 -y
conda activate dlpm
python -m pip install --upgrade pip
```

### 2. Install PyTorch

Install the PyTorch + torchvision build that matches your hardware directly from [pytorch.org](https://pytorch.org/get-started/locally/). For example, CUDA 12.1 users can run:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the Python dependencies

Most requirements are listed in `bem/requirements.txt`, so after PyTorch is installed run:

```bash
pip install -r bem/requirements.txt
```

Some optional components live outside that list:

- `pip install torchquad imageio tensorboard` (needed for the Lévy samplers, TinyImageNet loader, and LIM/TensorBoard logging)
- `pip install wandb` and run `wandb login` (or set `WANDB_API_KEY`) if you want remote logging; configure `WANDB_PROJECT`, `WANDB_ENTITY`, etc. as needed for `dlpm/NeptuneLogger.py`.
- `pip install torchlevy` when you plan to run the LIM baseline (`--method lim` or the scripts inside `dlpm/methods/LIM`).

### 4. Data preparation

`bem/datasets/__init__.py` expects datasets to live under `./data`. Torchvision datasets such as MNIST, CIFAR-10, CelebA, etc. will be downloaded automatically the first time you train. Toy 2‑D datasets are generated on the fly (`data/toy` already stores the masks). TinyImageNet must be downloaded manually into `data/tiny-imagenet-200`.

### 5. Optional LIM CUDA extensions

When you run LIM (`--method lim` or `dlpm/methods/LIM/main.py`), custom CUDA/CPP operators from `dlpm/methods/LIM/op/*.cu` are JIT-compiled via `torch.utils.cpp_extension.load`. Make sure the CUDA toolkit and a compatible compiler are in `PATH`; the first run may take a minute to build the extensions.

### 6. Quick smoke test

You can verify that everything is wired up by launching a tiny 2‑D experiment (no dataset downloads needed):

```bash
python run.py --config 2d_data --name smoke_test --method dlpm --epochs 1 --eval 1 --check 1 --n_max_batch 2 --train_reverse_steps 10
```

This should create checkpoints under `models/smoke_test/`.

### 7. Typical training & evaluation flow

Train (example on MNIST, see `dlpm/configs/mnist.yml` for parameters):

```bash
python run.py --config mnist --name dlpm_mnist --method dlpm --epochs 100 --eval 50 --check 50 --train_reverse_steps 1000
```

Evaluate/generate from the saved checkpoints:

```bash
python eval.py --config mnist --name dlpm_mnist --method dlpm --epochs 100 --eval 100 --generate 2000 --reverse_steps 1000
```

Both scripts must be run from the repository root so that the in-tree `bem` and `dlpm` packages can be imported without modifying `PYTHONPATH`.

## Examples

Below, we showcase side-by-side comparisons of images generated by DLPM ($\alpha = 1.7$):

| MNIST (DLPM) | CIFAR10_LT (DLPM) |
|-------------|------------------|
| <img src="./img/mnist_dlpm_1.7.png" alt="DLPM" width="300"/> | <img src="./img/cifar10_lt_dlpm_1.7.png" alt="LIM" width="300"/> |
---

And here is a comparison between DLPM and [LIM](https://openreview.net/forum?id=0Wp3VHX0Gm) across different numbers of reverse steps:

![DLPM vs. LIM](./img/dlpm_vs_lim_steps.png)



## Citation

```bibtex
@misc{shariatian2024denoisinglevyprobabilisticmodels,
      title={Denoising L\'evy Probabilistic Models}, 
      author={Dario Shariatian and Umut Simsekli and Alain Durmus},
      year={2024},
      eprint={2407.18609},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.18609}, 
}
```

## Contribute

We welcome issues, pull requests and contributions. We will try our best to improve readability and answer questions.
