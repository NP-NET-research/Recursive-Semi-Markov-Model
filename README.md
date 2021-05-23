# Recursive Semi CRF

This is the source code for "N-ary Constituent Tree Parsing with Recursive Semi-CRF" from ACL 2021.

`src_cn` and `src_en` are the codes used in Chinese and English tasks respectively.
## Contents
1. [Requirements](#Requirements)
2. [Training](#Training)
3. [Citation](#Citation)
4. [Credits](#Credits)

## Requirements

* Python : 3.7 or higher.
* Cython : 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) : 1.6.0 or any compatible version.
* [transformers](https://github.com/huggingface/pytorch-transformers) : 3.0 or any compatible version.
* [EVALB](http://nlp.cs.nyu.edu/evalb/) : run `make` in `./EVALB` first to compile `evalb` for evaluation.

## Training
Training requires cloning this repository from GitHub. You can download `src_cn` or 

### Training Instructions

Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
` --train-path` | Path to training constituent parsing | `data/02-21.10way.clean`
`--dev-path` | Path to development constituent parsing | `data/22.auto.clean`
`--batch-size` | Number of examples per training update | 250
`--subbatch-max-tokens` | Maximum number of words to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 100
`--checks-per-epoch` | Number of development evaluations per epoch | 4

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--use-tags` (for boolean parameters that default to False), or `--no-partitioned` (for boolean parameters that default to True).

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test constituent parsing | `data/23.auto.clean`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100

## Citation
pass

## Credits

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser
