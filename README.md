# Model-Agnostic Meta-Learning with transformers

Model-Agnostic Meta-Learning applied on layers of transformers.

### Dependencies
This code requires the following:
* cuda 9
* Python >= 3.5
* Tensorflow >= 1.10

### Data
ARSC Amazon Customer Reviews Sentiment Classification<br>
The dataset comprises reviews for 23 types of products on Amazon, and for each domain, there are 3 different binary classification tasks, so in total there are 69 different tasks. 12 of them are used for meta-testing and the rest are for meta-training.

### Word Embedding
Bert Word Embedding [bert-as-service](https://github.com/hanxiao/bert-as-service)<br>
Install Bert service<br>
Download pretrained bert model into `checkpoint` folder<br>
Run bert service on a seperate terminal process. The following command will return word embedding of each word in the sentence after padding.
```
bert-serving-start -model_dir ./MAML_transformer/checkpoint/uncased_L-12_H-768_A-12 -num_worker=4 -max_seq_len=50 -pooling_strategy NONE
```

### Usage
Run `main.py`.
