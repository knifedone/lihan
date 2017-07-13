from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import sys
model_path = sys.path[0] + '/simple_mnist.ckpt'
def compute_accuracy(v_xs, v_ys):
    global prediction1
    y_pre1 = sess.run(prediction1, feed_dict={xs: v_xs})
    correct_prediction1 = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    result1 = sess.run(accuracy1, feed_dict={xs: v_xs, ys: v_ys})
    return result1


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])


prediction1=tf.layers.dense( inputs=xs,units=10,
                            activation=tf.nn.softmax,
                            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                            bias_initializer=tf.constant_initializer(0.1),
                            name='fc1_1')

with tf.name_scope('loss1'):
    cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction1),
    reduction_indices=[1])) # loss

# train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy1)
#
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# writer = tf.summary.FileWriter("logs/", sess.graph)
# saver = tf.train.Saver()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys})
#     if i % 50 == 0:
#         print(compute_accuracy(mnist.test.images, mnist.test.labels))
# save_path = saver.save(sess, model_path)
# print ("[+] Model saved in file: %s" % save_path)



# train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver()

load_path=saver.restore(sess,model_path)
print ("[+] Model restored from %s" % load_path)
print(compute_accuracy(mnist.test.images, mnist.test.labels))