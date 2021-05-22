# Recursive Semi CRF

This is the source code of "N-ary Constituent Tree Parsing with Recursive Semi-CRF" from ACL 2019.

## Contents
1. [Requirements](#Requirements)
2. [Training](#Training)
3. [Citation](#Citation)
4. [Credits](#Credits)

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.0. This code has not been tested with PyTorch 1.0, but it should work.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. 
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) PyTorch 1.0.0+ or any compatible version (only required when using BERT and XLNet, XLNet only for joint span version.)

#### Pre-trained Models (PyTorch)

## Training

### Training Instructions

Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
` --train-ptb-path` | Path to training constituent parsing | `data/02-21.10way.clean`
`--dev-ptb-path` | Path to development constituent parsing | `data/22.auto.clean`
`--dep-train-ptb-path` | Path to training dependency parsing | `data/ptb_train_3.3.0.sd`
`--dep-dev-ptb-path` | Path to development dependency parsing | `data/ptb_dev_3.3.0.sd`
`--batch-size` | Number of examples per training update | 250
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--subbatch-max-tokens` | Maximum number of words to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 30
`--numpy-seed` | NumPy random seed | Random
`--use-words` | Use learned word embeddings | Do not use word embeddings
`--use-tags` | Use predicted part-of-speech tags as input | Do not use predicted tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-elmo` | Use pre-trained ELMo word representations | Do not use ELMo
`--use-bert` | Use pre-trained BERT word representations | Do not use BERT
`--use-xlnet` | Use pre-trained XLNet word representations | Do not use XLNet
`--pad-left` | When using pre-trained XLNet padding on left | Do not pad on left
`--bert-model` | Pre-trained BERT model to use if `--use-bert` is passed | `bert-large-uncased`
`--no-bert-do-lower-case` | Instructs the BERT tokenizer to retain case information (setting should match the BERT model in use) | Perform lowercasing
`--xlnet-model` | Pre-trained XLNet model to use if `--use-xlnet` is passed | `xlnet-large-cased`
`--no-xlnet-do-lower-case` | Instructs the XLNet tokenizer to retain case information (setting should match the XLNet model in use) | Perform uppercasing
`--const-lada` | Lambda weight | 0.5
`--model-name` | Name of model | test
`--embedding-path` | Path to pre-trained embedding | N/A
`--embedding-type` | Pre-trained embedding type | glove
`--dataset`     | Dataset type | ptb


Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--use-tags` (for boolean parameters that default to False), or `--no-partitioned` (for boolean parameters that default to True).

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-ptb-path` | Path to test constituent parsing | `data/23.auto.clean`
`--dep-test-ptb-path` | Path to test dependency parsing | `data/ptb_test_3.3.0.sd`
`--embedding-path` | Path to pre-trained embedding | `data/glove.6B.100d.txt.gz`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100
`--dataset`     | Dataset type | ptb

## Citation

## Credits

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser
