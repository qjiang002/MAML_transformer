
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from bert import modeling as bert_modeling

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 2-way classification).')
flags.DEFINE_integer('max_seq_len', 50, 'max sentence length')
## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 6000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-5, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 2e-6, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_bool('use_transformer', True, 'whether to use transformers to replace forward neural network')
flags.DEFINE_integer('num_transformer', 4, 'number of transformers')
## Model options
#flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
#flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
#flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
#flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_string("bert_config_file", './checkpoint/uncased_L-12_H-768_A-12/bert_config.json', "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string('meta_train_tasks','./data/meta_train_tasks.list','list of tasks in meta_training')
flags.DEFINE_string('meta_test_tasks','./data/meta_test_tasks.list','list of tasks in meta_testing')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './output', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 2
    SAVE_INTERVAL = 1000
    
    PRINT_INTERVAL = 2
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op] 

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)


        if itr % SUMMARY_INTERVAL == 0:
            
            prelosses.append(result[-4])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-3])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': loss: ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str+', accuracy: '+str(result[-2])+', '+str(result[-1]))
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            if model.classification:
                input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
            else:
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            
            result = sess.run(input_tensors, feed_dict)
            print('Validation accuracy results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
    print("training finished")
    #sess.close()

NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for itr in range(NUM_TEST_POINTS):
        
        feed_dict = {model.meta_lr : 0.0}
        
        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)

        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    meta_train_tasks = []
    with open(FLAGS.meta_train_tasks,'r') as f:
        for task in f:
            meta_train_tasks.append(task.strip())

    meta_test_tasks = []
    with open(FLAGS.meta_test_tasks,'r') as f:
        for task in f:
            meta_test_tasks.append(task.strip())

    #print("meta_train_tasks\n",meta_train_tasks)
    #print("meta_test_tasks\n",meta_test_tasks)


    bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    word_embedding_size = bert_config.hidden_size
    #print("sentence_embedding_size: ",sentence_embedding_size)

    if FLAGS.train == True:
        test_num_updates = 1  # eval on at least one update during training
    else:
        test_num_updates = 10
    
    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.train:
        data_generator = DataGenerator(FLAGS.meta_batch_size, meta_train_tasks, FLAGS.update_batch_size*2, word_embedding_size)
    else:
        data_generator = DataGenerator(FLAGS.meta_batch_size, meta_test_tasks, FLAGS.update_batch_size*2, word_embedding_size)
    
    dim_output = data_generator.dim_output
    #print("dim_output: ", dim_output)
    tf_data_load = True
    num_classes = data_generator.num_classes
    
    if FLAGS.train: # only construct training model if needed
        random.seed(5) 
        sentence_tensor, label_tensor = data_generator.make_data_tensor()
        inputa = tf.slice(sentence_tensor, [0,0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1, -1])
        inputb = tf.slice(sentence_tensor, [0,num_classes*FLAGS.update_batch_size, 0, 0], [-1,-1,-1, -1])
        labela = tf.slice(label_tensor, [0,0], [-1,num_classes*FLAGS.update_batch_size])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size], [-1,-1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    
    random.seed(6)
    sentence_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    inputa = tf.slice(sentence_tensor, [0,0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1, -1])
    inputb = tf.slice(sentence_tensor, [0,num_classes*FLAGS.update_batch_size, 0, 0], [-1,-1,-1, -1])
    labela = tf.slice(label_tensor, [0,0], [-1,num_classes*FLAGS.update_batch_size])
    labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size], [-1,-1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    
    model = MAML(word_embedding_size, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, bert_config=bert_config, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, bert_config=bert_config, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    tvars = tf.trainable_variables()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes) \
                +'.mbs_'+str(FLAGS.meta_batch_size) \
                + '.ubs_' + str(FLAGS.train_update_batch_size) \
                + '.numstep' + str(FLAGS.num_updates) \
                + '.updatelr' + str(FLAGS.train_update_lr) \
                + '.numtransformer' + str(FLAGS.num_transformer)
    
    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
