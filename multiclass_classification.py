import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def read_infile():
    mnist = input_data.read_data_sets("MNIST/data", one_hot=True)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def weights_biases_placeholder(n_dim, n_classes):
    x_p = tf.placeholder(tf.float32, [None, n_dim])
    y_p = tf.placeholder(tf.float32, [None, n_classes])
    w_p = tf.Variable(tf.random_normal([n_dim, n_classes], stddev=0.01), name="weights")
    b_p = tf.Variable(tf.random_normal([n_classes]), name="bias")
    return x_p, y_p, w_p, b_p


def forward_pass(w_param, b_param, x_param):
    return tf.matmul(x_param, w_param) + b_param


def multiclass_cost(out, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    return cost


def init():
    return tf.global_variables_initializer()


def training_op(learning_rate, cost):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Creating train test data, defining forward pass, cost functions
train_x, train_y, test_x, test_y = read_infile()
print("train x Shape: ", train_x.shape)
print("train y shape: ", train_y.shape)
X, Y, w, b = weights_biases_placeholder(train_x.shape[1], train_y.shape[1])
yhat = forward_pass(w, b, X)
costfn = multiclass_cost(yhat, Y)
learningrate, epochs = 0.01, 1000
op_train = training_op(learningrate, costfn)
init = init()
loss_trace = []
accuracy_trace = []
loss_ = 0
accuracy_ = 0

# Tensorflow Session
with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        sess.run(op_train, feed_dict={X: train_x, Y: train_y})
        loss_ = sess.run(costfn, feed_dict={X: train_x, Y: train_y})
        accuracy_ = np.mean(np.argmax(sess.run(yhat, feed_dict={X: train_x, Y: train_y}), axis=1)
                            == np.argmax(train_y, axis=1))
        loss_trace.append(loss_)
        accuracy_trace.append(accuracy_)
        if ((i + 1) >= 100) and ((i + 1) % 100 == 0):
            print("Epoch: ", (i + 1), "Loss: ", loss_, "accuracy: ", accuracy_)

    print("Final training result: ", "loss: ", loss_, "accuracy: ", accuracy_)
    loss_test = sess.run(costfn, feed_dict={X: test_x, Y: test_y})
    test_prediction = np.argmax(sess.run(yhat, feed_dict={X: test_x, Y: test_y}), axis=1)
    accuracy_test = np.mean(test_prediction == np.argmax(test_y, axis=1))
    print("Results on test dataset: ", "loss: ", loss_test, "accuracy: ", accuracy_test)

    print("Actual Digits: ", np.argmax(test_y[0:10], axis=1))
    print("Predicted Digits: ", test_prediction[0:10])
