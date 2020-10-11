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

from bert_serving.client import BertClient
bert_client = BertClient()


FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, batch_size, task_index_map,config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
            task_index_map
        """
        self.batch_size = batch_size
        #self.num_samples_per_class = num_samples_per_class
        #self.test_samples_per_class = test_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.task_index_map = task_index_map
        
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        #self.hidden_size = config.get('hidden_size', FLAGS.hidden_size)
        self.dim_output = self.num_classes
        metatrain_folder = config.get('metatrain_folder', './data/train')
        if FLAGS.test_set:
            metaval_folder = config.get('metaval_folder', './data/test')
        else:
            metaval_folder = config.get('metaval_folder', './data/dev')

        metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if label in self.task_index_map \
                ]

        metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if label in self.task_index_map \
                ]
        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders
        

    def make_data_tensor(self, num_samples_per_class, test_samples_per_class, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200
            print("num_total_batches: ", num_total_batches)
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        all_sentences = []
        all_labels = []
        all_test_sentences = []
        all_test_labels = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.batch_size)
            random.shuffle(sampled_character_folders)
            #print("sampled_character_folders: ", sampled_character_folders)
            sentence_embedding, labels, test_sentence_embedding, test_label_list = self.collect_batch_sentence_embedding(sampled_character_folders, num_samples_per_class, test_samples_per_class,)
            #print("sentence_embedding_shape: ",sentence_embedding.shape)
            #print("labels_shape: ",labels.shape)
            # make sure the above isn't randomized order
            all_sentences.append(sentence_embedding)
            all_labels.append(labels)
            all_test_sentences.append(test_sentence_embedding)
            all_test_labels.append(test_label_list)

        all_sentences_batches = tf.stack(all_sentences) #[num_total_batches, task_per_batch * num_class * sample_per_class, seq_len, embedding_size]
        all_label_batches = tf.stack(all_labels) 
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        all_test_sentences_batches = tf.stack(all_test_sentences) #[num_total_batches, task_per_batch * num_class * test_sample_per_class, seq_len, embedding_size]
        all_test_label_batches = tf.stack(all_test_labels) 
        all_test_label_batches = tf.one_hot(all_test_label_batches, self.num_classes)
        return all_sentences_batches, all_label_batches, all_test_sentences_batches, all_test_label_batches


    def collect_batch_sentence_embedding(self, sampled_character_folders, num_samples_per_class, test_samples_per_class,):
        support_set = []
        test_support_set = []
        for file in sampled_character_folders:
            task_support_set = []
            sample_pool = {'1': [], '-1':[]}
            with open(file, 'r') as f:
                for line in f:
                    text, label = line.strip().split('\t')
                    sample_pool[str(label)].append(text)
                #print(file, len(sample_pool['1']), len(sample_pool['-1']))
                pos_sample = random.sample(sample_pool['1'], num_samples_per_class)
                if num_samples_per_class > len(sample_pool['-1']) or num_samples_per_class < 1:
                    print("random.sample Error: ", file)
                    print("self.num_samples_per_class: ", num_samples_per_class)
                    print("len(sample_pool['-1']): ",len(sample_pool['-1']))
                neg_sample = random.sample(sample_pool['-1'], num_samples_per_class)
                task_support_set.extend([(s,1) for s in pos_sample]) 
                task_support_set.extend([(s,0) for s in neg_sample]) 
                random.shuffle(task_support_set)
                support_set.extend(task_support_set)

                pos_test_sample = random.sample(sample_pool['1'], test_samples_per_class)
                neg_test_sample = random.sample(sample_pool['-1'], test_samples_per_class)
                test_support_set.extend([(s,1) for s in pos_test_sample]) 
                test_support_set.extend([(s,0) for s in neg_test_sample]) 

        text_list = [s for (s,l) in support_set]
        label_list = [l for (s,l) in support_set]
        #print("label_list:\n", label_list)
        sentence_embedding = tf.convert_to_tensor(bert_client.encode(text_list)) #[task_per_batch * num_class * sample_per_class, seq_len, embedding_size]
        label_list = tf.convert_to_tensor(label_list) #task_per_batch * num_class * sample_per_class
        
        test_text_list = [s for (s,l) in test_support_set]
        test_label_list = [l for (s,l) in test_support_set]
        #print("label_list:\n", label_list)
        test_sentence_embedding = tf.convert_to_tensor(bert_client.encode(test_text_list)) #[task_per_batch * num_class * sample_per_class, seq_len, embedding_size]
        test_label_list = tf.convert_to_tensor(test_label_list) #task_per_batch * num_class * sample_per_class
        

        return sentence_embedding, label_list, test_sentence_embedding, test_label_list




