import os

import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from bert import modeling

FLAGS = flags.FLAGS

'''
for f in os.listdir('./data'):
	if f.endswith('.train'):
		os.rename('./data/'+f, './data/train/'+f.replace('.train',''))
	if f.endswith('.test'):
		os.rename('./data/'+f, './data/test/'+f.replace('.test',''))
	if f.endswith('.dev'):
		os.rename('./data/'+f, './data/dev/'+f.replace('.dev',''))

meta_train_list = []
meta_test_list = []
with open('./data/workspace.filtered.list','r') as f:
	for l in f:
		l=l.strip()
		meta_train_list.extend([l+'.t2', l+'.t4', l+'.t5'])
with open('./data/workspace.target.list','r') as f:
	for l in f:
		l=l.strip()
		meta_test_list.extend([l+'.t2', l+'.t4', l+'.t5'])

with open('./data/meta_train_tasks.list','w') as f:
	for t in meta_train_list:
		f.write(t+'\n') 

with open('./data/meta_test_tasks.list','w') as f:
	for t in meta_test_list:
		f.write(t+'\n') 

dic = {}
for file in os.listdir('./data/test'):
	domain = file.split('.')[0]
	if domain not in dic:
		dic[domain] = []
	with open('./data/test/'+file) as f:
		for line in f:
			dic[domain].append(line)
print(len(dic))
for (domain, f) in dic.items():
	with open('./data/test_domain/'+domain+'.txt','w+') as w:
		for line in f:
			w.write(line)
'''
tasks = []
with open('./data/meta_train_tasks.list','r') as f:
	for line in f:
		tasks.append(line.strip())
print(tasks)
data_generator = DataGenerator(
	meta_batch_size=4, 
	tasks=tasks, 
	num_classes=2, 
	support_samples_per_class=5, 
	query_samples_per_class=15,
	max_seq_len=128, 
	vocab_file='checkpoint/uncased_L-12_H-768_A-12/vocab.txt', 
	config={}, 
	test=False)
single_batch_input_ids, single_batch_segment_ids, single_batch_input_mask, single_batch_labels = data_generator.generate_single_batch_data()
print(single_batch_input_ids.shape)
print(single_batch_input_ids)
print(single_batch_segment_ids.shape)
print(single_batch_segment_ids)
print(single_batch_input_mask.shape)
print(single_batch_input_mask)
print(single_batch_labels.shape)
print(single_batch_labels)