""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import os
import math
import csv
import pickle
import random
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    from bert import modeling
    from bert import optimization
    from bert import tokenization
    import tensorflow as tf
    from sklearn.metrics import f1_score,precision_score,recall_score
    from tensorflow.python.ops import math_ops

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 2-way classification).')
## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 60000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_integer('max_seq_len', 128, 'max sentence length')
## Model options
#flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
#flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
#flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
#flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_string("bert_config_file", './checkpoint/MAML_bert_config.json', "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string('meta_train_tasks','./data/meta_train_tasks.list','list of tasks in meta_training')
flags.DEFINE_string('meta_test_tasks','./data/meta_test_tasks.list','list of tasks in meta_testing')
flags.DEFINE_string("vocab_file", 'checkpoint/uncased_L-12_H-768_A-12/vocab.txt', "BERT vocab file.")
flags.DEFINE_string("output_dir", 'output', "output directory.")

flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './output', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def main():
	tasks = []
	with open(FLAGS.meta_train_tasks,'r') as f:
		for line in f:
			tasks.append(line.strip())
	print('tasks: ', tasks)
	data_generator = DataGenerator(
						meta_batch_size=FLAGS.meta_batch_size, 
						tasks=tasks, 
						num_classes=FLAGS.num_classes, 
						support_samples_per_class=FLAGS.update_batch_size, 
						query_samples_per_class=15,
						max_seq_len=FLAGS.max_seq_len, 
						vocab_file=FLAGS.vocab_file, 
						config={}, 
						test=False)
	
	SUMMARY_INTERVAL = 100
	SAVE_INTERVAL = 500
	PRINT_INTERVAL = 1000
	TEST_PRINT_INTERVAL = 10

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement,
			log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			print("initializing model...")
			model = MAML(config_file=FLAGS.bert_config_file, 
				meta_batch_size=FLAGS.meta_batch_size, 
				support_samples_per_class=FLAGS.update_batch_size, 
				query_samples_per_class=15, 
				max_seq_len=FLAGS.max_seq_len, 
				meta_lr=FLAGS.meta_lr, 
				update_lr=FLAGS.update_lr, 
				num_updates=FLAGS.num_updates,
				num_classes=FLAGS.num_classes)
			print("constructing model...")
			model.construct_model(is_training=True)
			print("model done")
			model.summ_op = tf.summary.merge_all()
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(tf.global_variables())
			tvars = tf.trainable_variables()
			for var in tvars:
				print("name =  %s, shape = %s", var.name, var.shape)
			print('starting training...')
			for itr in range(FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
				
				print("itr: ", itr)
				single_batch_input_ids, single_batch_segment_ids, single_batch_input_mask, single_batch_labels = data_generator.generate_single_batch_data()
				
				input_ids_support = single_batch_input_ids[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
				input_ids_query = single_batch_input_ids[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
				segment_ids_support = single_batch_segment_ids[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
				segment_ids_query = single_batch_segment_ids[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
				input_mask_support = single_batch_input_mask[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
				input_mask_query = single_batch_input_mask[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
				labels_support = single_batch_labels[:, 0:FLAGS.num_classes*FLAGS.update_batch_size]
				labels_query = single_batch_labels[:, FLAGS.num_classes*FLAGS.update_batch_size:]
				#print(model.weights['output_weight'].eval())
				
				feed_dic = {
						model.input_ids_support : input_ids_support,
						model.input_ids_query : input_ids_query,
						model.segment_ids_support : segment_ids_support,
						model.segment_ids_query : segment_ids_query,
						model.input_mask_support : input_mask_support,
						model.input_mask_query : input_mask_query,
						model.labels_support : labels_support,
						model.labels_query : labels_query
					}
				input_tensors = [model.metatrain_op]
				input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
				input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
				result = sess.run(input_tensors, feed_dic)
				print('pre-loss: '+str(result[2])+', post-loss: '+str(result[3])+', pre-acc: '+str(result[4])+', post-acc:'+str(result[5]))
				#print(model.weights['output_weight'].eval())
				if (itr!=0) and itr % SAVE_INTERVAL == 0:
					saver.save(sess, FLAGS.logdir + '/model' + str(itr))

				if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
					single_batch_input_ids, single_batch_segment_ids, single_batch_input_mask, single_batch_labels = data_generator.generate_single_batch_data()
				
					input_ids_support = single_batch_input_ids[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
					input_ids_query = single_batch_input_ids[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
					segment_ids_support = single_batch_segment_ids[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
					segment_ids_query = single_batch_segment_ids[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
					input_mask_support = single_batch_input_mask[:, 0:FLAGS.num_classes*FLAGS.update_batch_size, :]
					input_mask_query = single_batch_input_mask[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
					labels_support = single_batch_labels[:, 0:FLAGS.num_classes*FLAGS.update_batch_size]
					labels_query = single_batch_labels[:, FLAGS.num_classes*FLAGS.update_batch_size:]

					val_feed_dic = {
						model.input_ids_support : input_ids_support,
						model.input_ids_query : input_ids_query,
						model.segment_ids_support : segment_ids_support,
						model.segment_ids_query : segment_ids_query,
						model.input_mask_support : input_mask_support,
						model.input_mask_query : input_mask_query,
						model.labels_support : labels_support,
						model.labels_query : labels_query
					}
					val_input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1], model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
					val_result = sess.run(val_input_tensors, val_feed_dic)
					print('Validation loss results: ' + str(val_result[0]) + ', ' + str(val_result[1]))
					print('Validation accuracy results: ' + str(val_result[2]) + ', ' + str(val_result[3]))


if __name__ == "__main__":
    main()