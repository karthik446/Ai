import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

from tensorflow.python.training.input import batch

mnist = input_data.read_data_sets("MNIST/data", one_hot=True)

# Parameters
learning_rate = 0.01
epochs = 20
batch_size = 256
num_batches = mnist.train.num_examples / batch_size
input_height = 28
input_width = 28
n_classes = 10
dropout = 0.75
display_step = 1  # what is this ?
filter_height = 5
filter_width = 5
depth_input = 1
depth_out1 = 64
depth_out2 = 128

# defining X Y
x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# defining weights
# wc-1 : filter_height*filter_width*depth_input*depth_out1 - number of weights
# wc-2: filter_height*filter_width*depth_out1*depth_out2
# wd-1: after max pooling the image size reduces to 1/16 i.e. input_height/4*input_width/4
# Output layer- 1024, n_classification_classes
weights = {
    'wc1': tf.Variable(tf.random_normal([filter_height, filter_width, depth_input, depth_out1])),
    'wc2': tf.Variable(tf.random_normal([filter_height, filter_width, depth_out1, depth_out2])),
    'wd1': tf.Variable(tf.random_normal([int((input_height / 4) * (input_width / 4) * depth_out2), 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

# defining biases
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# functions
# does conv2d, adds bias and returns relu of the input
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# pooling layer
def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


# create model
def conv_net(x_input, weights, biases, dropout):
    x_input = tf.reshape(x_input, shape=[-1, input_height, input_width, 1])
    # convolution layer 1
    conv1 = conv2d(x_input, weights["wc1"], biases["bc1"])
    conv1 = maxpool2d(conv1, 2)

    # convolution layer 2
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    conv2 = maxpool2d(conv2, 2)

    # fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases["bd1"])
    fc1 = tf.nn.relu(fc1)

    # apply drop out
    fc1 = tf.nn.dropout(fc1, dropout)

    # output prediction model
    out = tf.add(tf.matmul(fc1, weights["out"]), biases["out"])
    return out


# doing predictions
pred = conv_net(x, weights, biases, keep_prob)
# loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# accuracy
correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# init
init = tf.global_variables_initializer()

print(num_batches)
# execution graph
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(int(num_batches)):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            if epochs % display_step == 0:
                print("Epoch:", '%04d' % (i + 1),
                      "cost=", "{:.9f}".format(loss),
                      "Training accuracy", "{:.5f}".format(acc))
    print('Optimization Completed')

    y1 = sess.run(pred, feed_dict={x: mnist.test.images[:256], keep_prob: 1})
    test_classes = np.argmax(y1, 1)
    print('Testing Accuracy:',
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1}))
    f, a = plt.subplots(1, 10, figsize=(10, 2))

    for i in range(10):
        a[i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        print(test_classes[i])

end_time = time.time()
print('Total processing time:', end_time - start_time)