"""
File to reproduce the results for RNN-SPN

"""
from __future__ import division
import tensorflow as tf
import numpy as np
from math import ceil,floor
import argparse
from spatial_transformer import transformer
from tf_utils import drop_wrapper,max_pool_wrapper,dense_to_one_hot,RNN,nn_layer,tfrepeat,conv2d,\
					weight_variable,variable_summaries
from PIL import Image     

def save_as_image( img ,iter_i,epoch_i,which_img,type):
    img = img.reshape((img.shape[0],img.shape[1]))
    img = img*256
    contenated_image = Image.fromarray(img.astype(np.uint8),'L')
    contenated_image.save('try/'+'img'+' '+str(epoch_i) +'_' + str(iter_i)+'_'+ which_img + type+ '.bmp')
    return 

parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=str, default="0.0005")
parser.add_argument("-decayinterval", type=int, default=10)
parser.add_argument("-decayfac", type=float, default=1.5)
parser.add_argument("-nodecay", type=int, default=30)
parser.add_argument("-optimizer", type=str, default='rmsprop')
parser.add_argument("-dropout", type=float, default=0.0)
parser.add_argument("-downsample", type=float, default=3.0)
args = parser.parse_args()
mnist_sequence = "mnist_sequence3_sample_8distortions_9x9.npz"
data = np.load(mnist_sequence)

np.random.seed(123)
TOL = 1e-5
num_batch = 100
dim = 100
num_rnn_units = 256
num_classes = 10
NUM_EPOCH = 300
LR = float(args.lr)
MONITOR = False
MAX_NORM = 5.0
LOOK_AHEAD = 50



x_train, y_train = data['X_train'].reshape((-1, dim, dim)), data['y_train']
x_valid, y_valid = data['X_valid'].reshape((-1, dim, dim)), data['y_valid']
x_test, y_test = data['X_test'].reshape((-1, dim, dim)), data['y_test']
num_steps = y_train.shape[1]
batches_train = x_train.shape[0] // num_batch
batches_valid = x_valid.shape[0] // num_batch

y_train = y_train.reshape((-1))
y_valid = y_valid.reshape((-1))
y_test = y_test.reshape((-1))

y_train = dense_to_one_hot(y_train, n_classes=num_classes)
y_train = y_train.reshape(-1,num_steps,num_classes)

y_valid = dense_to_one_hot(y_valid, n_classes=num_classes)
y_valid = y_valid.reshape(-1,num_steps,num_classes)

y_test = dense_to_one_hot(y_test, n_classes=num_classes)
y_test = y_test.reshape(-1,num_steps,num_classes)

with tf.name_scope('learning_rate'):
	lr = tf.placeholder(tf.float32)
	tf.scalar_summary('learning_rate', lr)
with tf.name_scope('keep_prob'):
	keep_prob = tf.placeholder(tf.float32)
	tf.scalar_summary('keep_prob', keep_prob)

x = tf.placeholder(tf.float32, [None,dim,dim])
y = tf.placeholder(tf.float32, [None, num_classes])
X = tf.reshape(x, [-1, dim, dim, 1])
#define net
l_pool0_loc = max_pool_wrapper(X, 2, padding = 'VALID', name='l_pool0_loc')
l_conv0_loc = conv2d(l_pool0_loc, 20,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv0_loc")

l_pool1_loc = max_pool_wrapper(l_conv0_loc, 2, padding = 'VALID', name='l_pool1_loc')
l_conv1_loc = conv2d(l_pool1_loc, 20,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv1_loc")

l_conv1_loc_drop = drop_wrapper(l_conv1_loc,keep_prob,'l_con1_loc_drop')

l_pool2_loc = max_pool_wrapper(l_conv1_loc_drop, 2, padding = 'VALID', name='l_pool2_loc')
l_conv2_loc = conv2d(l_pool2_loc, 20,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv2_loc")

l_rnn_loc = RNN(l_conv2_loc,num_steps,num_rnn_units)

with tf.name_scope('fc_loc22'):
    with tf.name_scope('weights'):
        W_fc_loc22 = weight_variable([num_rnn_units, 6],method = 'zeros')
        variable_summaries(W_fc_loc22, 'fc_loc22' + '/weights')

# Use identity transformation as starting point
    with tf.name_scope('biases'):
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc22 = tf.Variable(initial_value=initial, name='b_fc_loc22')
        variable_summaries(l_rnn_loc, 'fc_loc22' + '/biases')

    with tf.name_scope('Wx_plus_b'):
      fc_loc = tf.matmul(l_rnn_loc, W_fc_loc22) + b_fc_loc22
      tf.histogram_summary('fc_loc' + '/out',fc_loc)     

downsampled_size = int(ceil(dim/args.downsample))    
out_size = (downsampled_size,downsampled_size,1)
repeated = tfrepeat(X,num_steps)
repeated = tf.reshape(repeated, [num_steps,-1,100,100,1])
repeated = tf.transpose(repeated, [1,0,2,3,4])
repeated = tf.reshape(repeated, [-1,100,100,1])

print('batches_train:' + str(batches_train))

h_trans = transformer(repeated, fc_loc, out_size)
l_conv0_out = conv2d(h_trans, 32,n_in = 1,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv0_out")
l_pool0_out = max_pool_wrapper(l_conv0_out, 2, padding = 'VALID', name='l_pool0_out')
l_pool0_out_drop = drop_wrapper(l_pool0_out,keep_prob,'l_pool0_out_drop')

l_conv1_out = conv2d(l_pool0_out_drop, 32,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv1_out")
l_pool1_out = max_pool_wrapper(l_conv1_out, 2, padding = 'VALID', name='l_pool1_out')
l_pool1_out_drop = drop_wrapper(l_pool1_out,keep_prob,'l_pool1_out_drop')

l_conv2_out = conv2d(l_pool1_out_drop, 32,k_h=3, k_w=3,stride_h=1, stride_w=1,name="l_conv2_out")
l_conv2_flat = tf.reshape(l_conv2_out, [-1, 5*5*32])

h_fc0_out = nn_layer(l_conv2_flat, 5*5*32, 400, 'fcnn_0')
y_logits = nn_layer(h_fc0_out, 400, num_classes, 'fcnn_1')


with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
    opt = tf.train.RMSPropOptimizer(learning_rate = lr)
    optimizer = opt.minimize(cross_entropy)

# %% Monitor accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.scalar_summary('accuracy', accuracy)

# %% We now create a new session to actually perform the initialization the
# variables:
merged = tf.merge_all_summaries()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
train_writer = tf.train.SummaryWriter('tensorboard_out' + '/train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard_out'+'/test',sess.graph)
print sess.run(tf.initialize_all_variables())

# %% We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = num_batch
n_epochs = 500
last_decay = 0
for epoch_i in range(NUM_EPOCH):
	shuffle = np.random.permutation(x_train.shape[0])
	#TODO decay drop rate
	for i in range(num_batch):
	    idx = shuffle[i*batches_train:(i+1)*batches_train]   
	   
	    batch_xs = x_train[idx]
	    batch_ys = y_train[idx].reshape((-1,num_classes)) 
	    if(i==0):		
	        
	        img = sess.run(X,
	                        feed_dict={
	                            x: batch_xs,
	                            y: batch_ys,
	                            keep_prob: 1.0,
	                            lr: LR
	                        })

	        save_as_image( img[0,:,:,:] ,i,epoch_i,'', 'origin')    


	        img = sess.run(h_trans,
	                        feed_dict={
	                            x: batch_xs,
	                            y: batch_ys,
	                            keep_prob: 1.0,
	                            lr: LR
	                        })
	        save_as_image( img[0,:,:,:] ,i,epoch_i,'0','transformed')   
	        save_as_image( img[1,:,:,:] ,i,epoch_i,'1','transformed')   
	        save_as_image( img[2,:,:,:] ,i,epoch_i,'2','transformed')   

	        img = sess.run(l_conv2_out,
	                        feed_dict={
	                            x: batch_xs,
	                            y: batch_ys,
	                            keep_prob: 1.0,
	                            lr: LR
	                        })
	        print img.shape	        
	        summary,loss = sess.run([merged,cross_entropy],
	                        feed_dict={
	                            x: batch_xs,
	                            y: batch_ys,
	                            keep_prob: 1.0,
	                            lr: LR
	                            })
	        train_writer.add_summary(summary, epoch_i*iter_per_epoch + i)
	        print('Iteration: ' + str(i) + ' Loss: ' + str(loss))

	    sess.run(optimizer, feed_dict={
	        x: batch_xs, y: batch_ys, keep_prob: 1.0,lr: LR})
	        #train_writer.add_summary(summary, epoch_i*iter_per_epoch + iter_i)
	if last_decay > args.decayinterval and epoch_i > args.nodecay:
	    last_decay = 0
	    old_lr = LR
	    LR = LR / args.decayfac
	    print "Decay lr from %f to %f" % (float(old_lr), float(LR))
	else:
	    last_decay += 1    
	acc,summary = sess.run([accuracy,merged],
	      feed_dict={
	      x: x_valid[0:1000],
	      y: y_valid[0:1000].reshape((-1,num_classes)),
	      keep_prob: 1.0,
	      lr: LR
	                  })
	test_writer.add_summary(summary, epoch_i*iter_per_epoch + i)
	print('Accuracy (%d): ' % epoch_i + str(acc))

    # theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    # print(theta[0])