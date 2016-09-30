# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
import tensorflow as tf
import numpy as np

def conv2d(x, n_filters,
           n_in = 0,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=lambda x: x,
           bias=False,
           padding='VALID',
           name="Conv2D"):
    """2D Convolution with options for kernel size, stride, and init deviation.
    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        with tf.name_scope('weights'):
            if(n_in == 0):
                w = tf.get_variable(
                    'w', [k_h, k_w, x.get_shape()[-1], n_filters],
                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                w = tf.get_variable(
                    'w', [k_h, k_w, n_in, n_filters],
                    initializer=tf.contrib.layers.xavier_initializer())
            variable_summaries(w, name + '/weights')
        with tf.name_scope('conv'):        
            conv = tf.nn.conv2d(
                x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            with tf.name_scope('biases'):
                b = tf.get_variable(
                    'b', [n_filters],
                initializer=tf.contrib.layers.xavier_initializer())
                variable_summaries(b, name + '/bias')
            with tf.name_scope('conv'):    
                conv = conv + b
                    
        with tf.name_scope('conv'):        
            tf.histogram_summary(name + '/conv', conv)   
        return conv
def drop_wrapper(x,keep_prob,name):
    with tf.variable_scope(name):
        droped = tf.nn.dropout(x, keep_prob)
    tf.histogram_summary(name + '/droped', droped)
    return droped
def max_pool_wrapper(x, k, padding = 'VALID', name='Maxpool'):
    with tf.variable_scope(name):
            pool = tf.nn.max_pool(x, [1,k,k,1],  [1,k,k,1], padding)
    tf.histogram_summary(name + '/pool', pool)       
    return pool
def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
    """Fully-connected network.
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    scope : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))
    
# %%
def weight_variable(shape,method = 'xavier',name = ''):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.00)
    if method == "zeros":
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
    

# %%
def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.00001)
    return tf.Variable(initial)

# %% 
def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)



def RNN(a,num_steps,num_rnn_units):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    with tf.variable_scope('Rnn'):

        shape = a.get_shape()
        map_size = shape[1]*shape[2]*shape[3]
        x = tfrepeat(a,num_steps)

        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, int(map_size)])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, num_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_rnn_units, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, num_rnn_units])
        # Linear activation, using rnn inner loop last output
        with tf.name_scope('rnn'):        
            tf.histogram_summary('Rnn' + '/rnn', outputs) 
    return outputs

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu,method = "xavier"):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim],method = method,name = layer_name)
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights)
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    if act is None:
      activations = preactivate
    else:
      activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def tfrepeat(a, repeats):
  return tf.tile(a, [repeats,1,1,1 ])

