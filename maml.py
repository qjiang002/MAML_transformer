""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import math
import copy
import six
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
from bert import modeling


FLAGS = flags.FLAGS

class MAML:
    def __init__(self, config_file, meta_batch_size, support_samples_per_class, query_samples_per_class, max_seq_len, meta_lr, update_lr, num_updates, num_classes):
        """ must call construct_model() after initializing MAML! """
        self.config = modeling.BertConfig.from_json_file(config_file)
        self.meta_batch_size = meta_batch_size
        self.update_lr = tf.placeholder_with_default(update_lr, ())
        self.meta_lr = tf.placeholder_with_default(meta_lr, ())
        self.support_samples_per_class = support_samples_per_class
        self.query_samples_per_class = query_samples_per_class
        self.num_updates = num_updates
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        #self.loss_func = xent

        self.forward = self.forward_transformer
        self.construct_weights = self.construct_transformer_weights
        self.loss_func = self.loss_func_transformer



    def construct_model(self, is_training=True):
        # a: training data for inner gradient, b: test data for meta gradient
        

        #self.single_batch_input_ids = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len), name='single_batch_input_ids')
        #self.single_batch_segment_ids = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len), name='single_batch_segment_ids')
        #self.single_batch_input_mask = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*(self.support_samples_per_class+self.query_samples_per_class), self.max_seq_len), name='single_batch_input_mask')
        #self.single_batch_labels = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*(self.support_samplt_samples_per_class+self.query_sames_per_class+self.query_samples_per_class)), name='single_batch_labels')
        self.input_ids_support = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.support_samples_per_class, self.max_seq_len), name='input_ids_support')
        self.input_ids_query = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.query_samples_per_class, self.max_seq_len), name='input_ids_query')
        self.segment_ids_support = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.support_samples_per_class, self.max_seq_len), name='segment_ids_support')
        self.segment_ids_query = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.query_samples_per_class, self.max_seq_len), name='segment_ids_query')
        self.input_mask_support = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.support_samples_per_class, self.max_seq_len), name='input_mask_support')
        self.input_mask_query = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.query_samples_per_class, self.max_seq_len), name='input_mask_query')
        self.labels_support = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.support_samples_per_class), name='labels_support')
        self.labels_query = tf.placeholder(dtype=tf.int64, shape=(self.meta_batch_size, self.num_classes*self.query_samples_per_class), name='labels_query')
        
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights(config=self.config)

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            outputbs = [[]]*self.num_updates
            lossesb = [[]]*self.num_updates
            accuraciesb = [[]]*self.num_updates

            def task_metalearn(inp):
                #print("enter task_metalearn")
                """ Perform gradient descent for one task in the meta-batch. """
                #input_ids, segment_ids, input_mask, labels = inp
                task_outputbs, task_lossesb, task_accuraciesb = [], [], []
                input_ids_support, input_ids_query, segment_ids_support, segment_ids_query, input_mask_support, input_mask_query, labels_support, labels_query = inp
                '''
                input_ids_support = tf.slice(input_ids, [0,0], [self.num_classes*self.support_samples_per_class, -1])
                input_ids_query = tf.slice(input_ids, [self.num_classes*self.support_samples_per_class,0], [-1, -1])
                segment_ids_support = tf.slice(segment_ids, [0,0], [self.num_classes*self.support_samples_per_class, -1])
                segment_ids_query = tf.slice(segment_ids, [self.num_classes*self.support_samples_per_class,0], [-1, -1])
                input_mask_support = tf.slice(input_mask, [0,0], [self.num_classes*self.support_samples_per_class, -1])
                input_mask_query = tf.slice(input_mask, [self.num_classes*self.support_samples_per_class,0], [-1, -1])
                labels_support = tf.slice(labels, [0], [self.num_classes*self.support_samples_per_class])
                labels_query = tf.slice(labels, [self.num_classes*self.support_samples_per_class], [-1])
                '''

                task_outputa = self.forward(input_ids=input_ids_support, 
                                            input_mask=input_mask_support, 
                                            token_type_ids=segment_ids_support, 
                                            weights=weights, 
                                            config=self.config, 
                                            is_training=is_training)
                task_lossa = self.loss_func(task_outputa, labels_support)

                grads = tf.gradients(task_lossa, list(weights.values()))
                #if FLAGS.stop_grad:
                #    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()])) #temporary weights
                output = self.forward(input_ids=input_ids_query, 
                                            input_mask=input_mask_query, 
                                            token_type_ids=segment_ids_query,  
                                            weights=fast_weights, 
                                            config=self.config, 
                                            is_training=is_training)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labels_query))

                for j in range(self.num_updates - 1):
                    loss = self.loss_func(self.forward(input_ids=input_ids_support, 
                                            input_mask=input_mask_support, 
                                            token_type_ids=segment_ids_support, 
                                            weights=fast_weights, 
                                            config=self.config, 
                                            is_training=is_training), labels_support)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    #if FLAGS.stop_grad:
                    #    grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(input_ids=input_ids_query, 
                                            input_mask=input_mask_query, 
                                            token_type_ids=segment_ids_query, 
                                            weights=fast_weights, 
                                            config=self.config, 
                                            is_training=is_training)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labels_query))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(predictions=tf.argmax(task_outputa, axis=-1), labels=labels_support)
                for j in range(self.num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(predictions=tf.argmax(task_outputbs[j], axis=-1), labels=labels_query))
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output 

            #if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                #unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*self.num_updates, tf.float32, [tf.float32]*self.num_updates, tf.float32, [tf.float32]*self.num_updates]
            
            result = tf.map_fn(task_metalearn, elems=(self.input_ids_support, self.input_ids_query, self.segment_ids_support, self.segment_ids_query, self.input_mask_support, self.input_mask_query, self.labels_support, self.labels_query), dtype=out_dtype, parallel_iterations=self.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            

        ## Performance & Optimization
        if is_training:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.num_updates-1])
            self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(self.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]

        ## Summaries
        tf.summary.scalar('Pre-update loss', total_loss1)
        tf.summary.scalar('Pre-update accuracy', total_accuracy1)

        for j in range(self.num_updates):
            tf.summary.scalar('Post-update loss, step ' + str(j+1), total_losses2[j])
            tf.summary.scalar('Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    def loss_func_transformer(self, logits, label):
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(label, depth=self.num_classes, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss

    def construct_transformer_weights(self, config):
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size
        initializer_range = config.initializer_range
        vocab_size = config.vocab_size
        type_vocab_size = config.type_vocab_size
        max_position_embeddings = config.max_position_embeddings

        weights = {}
        dtype = tf.float32
        size_per_head = int(hidden_size/num_attention_heads)

        weights['word_embedding'] = tf.get_variable('word_embedding', [vocab_size, hidden_size], initializer=create_initializer(initializer_range))
        weights['token_type_embedding'] = tf.get_variable('token_type_embedding', [type_vocab_size, hidden_size], initializer=create_initializer(initializer_range))
        weights['position_embededing'] = tf.get_variable('position_embededing', [max_position_embeddings, hidden_size], initializer=create_initializer(initializer_range))
        

        for i in range(config.num_hidden_layers):
            transformer_name = 'transformer_'+str(i+1)
            weights[transformer_name+'_attention_layer_query_weight'] = tf.get_variable(transformer_name+'_attention_layer_query_weight', [hidden_size, num_attention_heads * size_per_head], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_attention_layer_query_bias'] = tf.get_variable(transformer_name+'_attention_layer_query_bias', [num_attention_heads * size_per_head], initializer=tf.zeros_initializer())
            weights[transformer_name+'_attention_layer_key_weight'] = tf.get_variable(transformer_name+'_attention_layer_key_weight', [hidden_size, num_attention_heads * size_per_head], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_attention_layer_key_bias'] = tf.get_variable(transformer_name+'_attention_layer_key_bias', [num_attention_heads * size_per_head], initializer=tf.zeros_initializer())
            weights[transformer_name+'_attention_layer_value_weight'] = tf.get_variable(transformer_name+'_attention_layer_value_weight', [hidden_size, num_attention_heads * size_per_head], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_attention_layer_value_bias'] = tf.get_variable(transformer_name+'_attention_layer_value_bias', [num_attention_heads * size_per_head], initializer=tf.zeros_initializer())
            weights[transformer_name+'_attention_output_weight'] = tf.get_variable(transformer_name+'_attention_output_weight', [num_attention_heads * size_per_head, hidden_size], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_attention_output_bias'] = tf.get_variable(transformer_name+'_attention_output_bias', [hidden_size], initializer=tf.zeros_initializer())
            weights[transformer_name+'_intermediate_weight'] = tf.get_variable(transformer_name+'_intermediate_weight', [hidden_size, intermediate_size], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_intermediate_bias'] = tf.get_variable(transformer_name+'_intermediate_bias', [intermediate_size], initializer=tf.zeros_initializer())
            weights[transformer_name+'_intermediate_output_weight'] = tf.get_variable(transformer_name+'_intermediate_output_weight', [intermediate_size, hidden_size], initializer=create_initializer(initializer_range))
            weights[transformer_name+'_intermediate_output_bias'] = tf.get_variable(transformer_name+'_intermediate_output_bias', [hidden_size], initializer=tf.zeros_initializer())
        
        weights['pooler_weight'] = tf.get_variable('pooler_weight', [hidden_size, hidden_size], initializer=create_initializer(initializer_range))
        weights['pooler_bias'] = tf.get_variable('pooler_bias', [hidden_size], initializer=tf.zeros_initializer())
        weights['output_weight'] = tf.get_variable('output_weight', [hidden_size, self.num_classes], initializer=create_initializer(initializer_range))
        weights['output_bias'] = tf.get_variable('output_bias', [self.num_classes], initializer=tf.zeros_initializer())
        return weights

    def forward_transformer(self, input_ids, input_mask, token_type_ids, weights, config, is_training): # inp = [batch_size, seq_length, hidden_size]
        #print("enter forward_transformer")
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0


        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        embedding_output = embedding_lookup(
            input_ids=input_ids,
            weights=weights,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=False)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = embedding_postprocessor(
            input_tensor=embedding_output,
            weights=weights,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)


        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        all_encoder_layers = transformer_model(
            input_tensor=embedding_output, #[batch_size, seq_length, hidden_size]
            weights=weights,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act), #gelu
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)
        
        sequence_output = all_encoder_layers[-1]
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1) #[batch_size, hidden_size]
        pooler_weight = weights['pooler_weight']
        pooler_bias = weights['pooler_bias']
        pooled_output = tf.tanh( tf.matmul( first_token_tensor,pooler_weight ) + pooler_bias )
        if is_training:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)
        output_weights = weights['output_weight']
        output_bias = weights['output_bias']
        logits = tf.matmul(pooled_output, output_weights)
        logits = tf.nn.bias_add(logits, output_bias)
        return logits #[batch_size, self.dim_output]

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def embedding_lookup(input_ids,
                     weights,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])


  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, weights['word_embedding'])
  else:
    output = tf.gather(weights['word_embedding'], flat_input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return output

def embedding_postprocessor(input_tensor,
                            weights,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, weights['token_type_embedding'])
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(weights['position_embededing'], [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  #output = layer_norm_and_dropout(output, dropout_prob)
  output = dropout(output, dropout_prob)
  return output

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def transformer_model(input_tensor,
                      weights,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=4,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn='gelu',
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    #with tf.variable_scope("layer_%d" % layer_idx):
    layer_input = prev_output
    transformer_name = 'transformer_'+str(layer_idx+1)

      #with tf.variable_scope("attention"):
    attention_heads = []
        #with tf.variable_scope("self"):
    attention_head = attention_layer(  #[batch_size, from_seq_length, num_attention_heads * size_per_head]
              from_tensor=layer_input,
              to_tensor=layer_input,
              weights=weights,
              transformer_name=transformer_name,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
    attention_heads.append(attention_head)

    attention_output = None
    if len(attention_heads) == 1:
        attention_output = attention_heads[0]
    else:
        # In the case where we have other sequences, we just concatenate
        # them to the self-attention head before the projection.
        attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        #with tf.variable_scope("output"):
    attention_output_weight = weights[transformer_name+'_attention_output_weight']
    attention_output_bias = weights[transformer_name+'_attention_output_bias']
    attention_output = tf.matmul( attention_output,attention_output_weight ) + attention_output_bias
    attention_output = dropout(attention_output, hidden_dropout_prob)
    #attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      #with tf.variable_scope("intermediate"):
    intermediate_weight = weights[transformer_name+'_intermediate_weight']
    intermediate_bias = weights[transformer_name+'_intermediate_bias']
    intermediate_output = gelu(tf.matmul( attention_output,intermediate_weight ) + intermediate_bias )

      # Down-project back to `hidden_size` then add the residual.
      #with tf.variable_scope("output"):
    intermediate_output_weight = weights[transformer_name+'_intermediate_output_weight']
    intermediate_output_bias = weights[transformer_name+'_intermediate_output_bias']
    layer_output = tf.matmul( intermediate_output,intermediate_output_weight ) + intermediate_output_bias 
    layer_output = dropout(layer_output, hidden_dropout_prob)
    #layer_output = layer_norm(layer_output + attention_output)
    
    prev_output = layer_output
    all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def attention_layer(from_tensor,
                    to_tensor,
                    weights,
                    transformer_name,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_weights = weights[transformer_name+'_attention_layer_query_weight']
  query_bias = weights[transformer_name+'_attention_layer_query_bias']
  query_layer = tf.matmul(from_tensor_2d, query_weights) + query_bias

  # `key_layer` = [B*T, N*H]
  key_weights = weights[transformer_name+'_attention_layer_key_weight']
  key_bias = weights[transformer_name+'_attention_layer_key_bias']
  key_layer = tf.matmul(to_tensor_2d, key_weights) + key_bias
  

  # `value_layer` = [B*T, N*H]
  value_weights = weights[transformer_name+'_attention_layer_value_weight']
  value_bias = weights[transformer_name+'_attention_layer_value_bias']
  value_layer = tf.matmul(to_tensor_2d, value_weights) + value_bias
  

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer
