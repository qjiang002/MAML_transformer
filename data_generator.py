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


class DataGenerator(object):
    
    def __init__(self, meta_batch_size, tasks, num_classes, support_samples_per_class, query_samples_per_class, max_seq_len, vocab_file, config={}, test=False):
        
        self.meta_batch_size = meta_batch_size
        self.support_samples_per_class = support_samples_per_class
        self.query_samples_per_class = query_samples_per_class
        self.num_classes = num_classes
        self.tasks = tasks
        self.max_seq_len = max_seq_len
        print("tasks: ", tasks)
        
        self.metatrain_folder_dir = config.get('metatrain_folder', './data/train_domain')
        if test:
            self.metaval_folder_dir = config.get('metaval_folder', './data/test_domain')
        else:
            self.metaval_folder_dir = config.get('metaval_folder', './data/dev_domain')

        
        metatrain_tasks_support = {}
        for task in self.tasks:
            metatrain_tasks_support[task] = {'-1':[], '1':[]}
            with open(os.path.join(self.metatrain_folder_dir, task+'.txt'),'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    metatrain_tasks_support[task][line[1]].append(line[0])
        metaval_tasks_query = {}
        for task in self.tasks:
            metaval_tasks_query[task] = {'-1':[], '1':[]}
            with open(os.path.join(self.metaval_folder_dir, task+'.txt'),'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    metaval_tasks_query[task][line[1]].append(line[0])

        self.metatrain_tasks_support = metatrain_tasks_support
        self.metaval_tasks_query = metaval_tasks_query
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


    def sentence_tokenization(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[0:(self.max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens:
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        return input_ids, segment_ids, input_mask

    def generate_single_batch_data(self, train=True):
        
        single_batch_input_ids = []
        single_batch_segment_ids = []
        single_batch_input_mask = []
        single_batch_labels = []

        sampled_tasks = random.sample(self.tasks, self.meta_batch_size)
        for task in sampled_tasks:
            task_input_ids = []
            task_segment_ids = []
            task_input_mask = []
            task_labels = []

            single_support_set = random.sample(self.metatrain_tasks_support[task]['1'], self.support_samples_per_class) + \
                            random.sample(self.metatrain_tasks_support[task]['-1'], self.support_samples_per_class)
            single_support_labels = [1]*self.support_samples_per_class + [0]*self.support_samples_per_class
            c = list(zip(single_support_set, single_support_labels))
            random.shuffle(c)
            single_support_set, single_support_labels = zip(*c)
            
            single_query_set = random.sample(self.metaval_tasks_query[task]['1'], self.query_samples_per_class) + \
                            random.sample(self.metaval_tasks_query[task]['-1'], self.query_samples_per_class)
            single_query_labels = [1]*self.query_samples_per_class + [0]*self.query_samples_per_class
            c = list(zip(single_query_set, single_query_labels))
            random.shuffle(c)
            single_query_set, single_query_labels = zip(*c)

            single_task_set = single_support_set + single_query_set
            task_labels = single_support_labels + single_query_labels

            for sentence in single_task_set:
                input_ids, segment_ids, input_mask = self.sentence_tokenization(sentence)
                task_input_ids.append(input_ids)
                task_segment_ids.append(segment_ids)
                task_input_mask.append(input_mask)

            single_batch_input_ids.append(task_input_ids)
            single_batch_segment_ids.append(task_segment_ids)
            single_batch_input_mask.append(task_input_mask)
            single_batch_labels.append(task_labels)

        single_batch_input_ids = np.array(single_batch_input_ids)
        single_batch_segment_ids = np.array(single_batch_segment_ids)
        single_batch_input_mask = np.array(single_batch_input_mask)
        single_batch_labels = np.array(single_batch_labels)

        assert single_batch_input_ids.shape == (self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len)
        assert single_batch_segment_ids.shape == (self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len)
        assert single_batch_input_mask.shape == (self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len)
        assert single_batch_labels.shape == (self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class))

        return single_batch_input_ids, single_batch_segment_ids, single_batch_input_mask, single_batch_labels