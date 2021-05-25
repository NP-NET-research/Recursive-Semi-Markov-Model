# Recursive Semi-Markov Model

This is the source code for "N-ary Constituent Tree Parsing with Recursive Semi-Markov Model" from ACL 2021.

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
Training requires cloning this repository from GitHub. You can download only one of `src_cn` and `src_en` for English or Chinese tasks.

### Training Instructions

A new model can be trained using the command `python main.py train ...` in directory `src_cn` or `src_cn`. 

The available arguments are listed in the training part at https://github.com/nikitakit/self-attentive-parser.

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

As an example, after setting the paths for data and embeddings, you can use the following command to train a model:
```
python main.py train 
    --use-bert 
    --model-path-base ../model/cn_bert_t1 
    --num-layers 2  
    --learning-rate 0.00001 
    --batch-size 32 
    --eval-batch-size 16 
    --subbatch-max-tokens 500
```

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python main.py test ...` in directory `src_cn` or `src_cn`.

As an example, you can use the following command to evaluate a trained model:
```
python main.py test --model-path-base ../model/cn_bert_t1_dev=*.pt
```

## Citation
If you are interested in our researches, please cite:
```
Xin Xin, Jinlong Li and Zeqi Tan. 2021. N-ary Constituent Tree Parsing with Recursive Semi-Markov Model. Accept by ACL-IJCNLP.
```

## Credits
Our work is based on [Constituency Parsing with a Self-Attentive Encoder](https://www.aclweb.org/anthology/P18-1249) and [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://www.aclweb.org/anthology/P19-1340). 

The code in this repository and portions of this README are based on https://github.com/nikitakit/self-attentive-parser
