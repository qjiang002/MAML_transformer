# Model-Agnostic Meta-Learning with transformers

Model-Agnostic Meta-Learning applied on layers of transformers.

### Dependencies
This code requires the following:
* cuda 9
* Python >= 3.5
* Tensorflow >= 1.10


### Data
ARSC Amazon Customer Reviews Sentiment Classification<br>
The dataset comprises reviews for 23 types of products on Amazon, and for each domain, there are 3 different binary classification tasks. The three tasks in each domain are combined so that there are enough data for each class, so in total there are 23 different tasks/domains. 18 domains in `data/meta_train_tasks.list` are used for meta-testing and 4 domains in `data/meta_test_tasks.list` are used for meta-training.

### Word Embedding
Bert Word Embedding<br>
Install Bert service [bert-as-service](https://github.com/hanxiao/bert-as-service)<br>
Download pretrained bert model [BERT_Base uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) into `checkpoint` folder<br>
Bert-server and Bert-client are called in `data_generator.py`

### Train
download [bert](https://github.com/google-research/bert)<br>
Run `main.py`.
