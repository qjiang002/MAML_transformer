""" Code for loading data. """
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
#from utils import get_images


import collections
import os
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    from bert import modeling
    # from bert import modeling as modeling
    from bert import optimization
    from bert import tokenization
    import tensorflow as tf
    from sklearn.metrics import f1_score,precision_score,recall_score
    from tensorflow.python.ops import math_ops
    #import tf_metrics
import pickle
import time
import sys
import json
from tensorflow.contrib import predictor
import pathlib
import collections
import random
import numpy as np
FLAGS = flags.FLAGS

from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
args = get_args_parser().parse_args(['-model_dir', './checkpoint/uncased_L-12_H-768_A-12',
                                     '-num_worker', '1',
                                     '-max_seq_len', '50',
                                     '-pooling_strategy', 'NONE'])
server = BertServer(args)
server.start()
bert_client = BertClient()


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, batch_size, tasks, num_samples_per_class, word_embedding_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
            task_index_map
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        #self.test_samples_per_class = test_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.tasks = tasks
        print("tasks: ", tasks)
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.word_embedding_size = word_embedding_size
        self.dim_output = self.num_classes
        metatrain_folder = config.get('metatrain_folder', './data/train_domain')
        if FLAGS.test:
            metaval_folder = config.get('metaval_folder', './data/test_domain')
        else:
            metaval_folder = config.get('metaval_folder', './data/dev_domain')

        metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if label.split('.')[0] in self.tasks \
                ]

        metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if label.split('.')[0] in self.tasks \
                ]
        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders
        

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 10
            print("num_total_batches: ", num_total_batches)
        else:
            folders = self.metaval_character_folders
            num_total_batches = 10

        all_sentences = []
        all_labels = []
        all_test_sentences = []
        all_test_labels = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.batch_size)
            random.shuffle(sampled_character_folders)
            sentence_embedding, labels = self.collect_batch_sentence_embedding(sampled_character_folders, self.num_samples_per_class)
            all_sentences.extend(sentence_embedding)
            all_labels.extend(labels)
            
        all_sentences = np.array(all_sentences)
        all_labels = np.array(all_labels)
        
        input_queue = tf.train.slice_input_producer([all_sentences, all_labels],shuffle=False)
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        batch_sentence_size = self.batch_size * self.num_classes * self.num_samples_per_class
        sentence_batch, label_batch = tf.train.batch(
                input_queue,
                batch_size = batch_sentence_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_sentence_size,
                )
        
        #label_batch = tf.one_hot(label_batch, self.num_classes)
        sentence_batch = tf.reshape(sentence_batch, [self.batch_size, self.num_classes * self.num_samples_per_class, FLAGS.max_seq_len, self.word_embedding_size])
        label_batch = tf.reshape(label_batch, [self.batch_size, self.num_classes * self.num_samples_per_class])
        
        return sentence_batch, label_batch


    def collect_batch_sentence_embedding(self, sampled_character_folders, num_samples_per_class):
        support_set = []
        for file in sampled_character_folders:
            task_support_set = []
            sample_pool = {'1': [], '-1':[]}
            with open(file, 'r') as f:
                for line in f:
                    text, label = line.strip().split('\t')
                    sample_pool[str(label)].append(text)
                #print(file, len(sample_pool['1']), len(sample_pool['-1']))
                if (len(sample_pool['-1']) < num_samples_per_class) or (len(sample_pool['1']) < num_samples_per_class):
                    print("num_samples_per_class > #samples in this file: ", file)
                pos_sample = random.sample(sample_pool['1'], num_samples_per_class)
                neg_sample = random.sample(sample_pool['-1'], num_samples_per_class)
                task_support_set.extend([(s,1) for s in pos_sample]) 
                task_support_set.extend([(s,0) for s in neg_sample]) 
                random.shuffle(task_support_set)
                support_set.extend(task_support_set)
                
        random.shuffle(support_set)
        text_list = [s for (s,l) in support_set]
        label_list = [l for (s,l) in support_set]
        #print("bert_client encoding")
        sentence_embedding = bert_client.encode(text_list) #[task_per_batch * num_class * sample_per_class, seq_len, embedding_size]
        

        return sentence_embedding, label_list




