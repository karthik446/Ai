import tensorflow as tf
import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


# load boston data set
def read_infile():
    data = load_boston()
    features_boston: ndarray = np.array(data.data)
    target_boston: ndarray = np.array(data.target)
    return features_boston, target_boston


# feature normalization
def feature_normalize(data: ndarray) -> ndarray:
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std


# append bias i.e. the b term in wx + b
def append_bias(features_norm, target_norm):
    n_samples = features_norm.shape[0]
    n_features = features_norm.shape[1]
    print("Num of samples %s" % n_samples)
    print("Num of features %s" % n_features)
    intercept_feature = np.ones((n_samples, 1))
    x_bias: ndarray = np.concatenate((features_norm, intercept_feature), axis=1)
    x_bias = np.reshape(x_bias, [n_samples, n_features + 1])
    y_bias: ndarray = np.reshape(target_norm, [n_samples, 1])
    return x_bias, y_bias


features, target = read_infile()
z_features = feature_normalize(features)
X_input, Y_input = append_bias(z_features, target)
num_features = X_input.shape[1]

# Creating Tensor flow placeholders for X, Y, weights

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.random_normal((num_features, 1)), name="Weights")
init = tf.global_variables_initializer()

# Define tensor flow constants
learning_rate = 0.01
num_epochs = 1000
cost_trace = []
prediction = tf.matmul(X, w)
error = prediction - Y
cost = tf.reduce_mean(tf.square(error))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epochs):
        sess.run(train_op, feed_dict={X: X_input, Y: Y_input})
        cost_trace.append(sess.run(cost, feed_dict={X: X_input, Y: Y_input}))
    error = sess.run(error, feed_dict={X: X_input, Y: Y_input})
    pred_ = sess.run(prediction, {X: X_input})

print('MSE in training: ', cost_trace[-1])
print(cost_trace)
plt.plot(cost_trace)

# plot predicted house prices vs actual prices

fig, ax = plt.subplots()
plt.scatter(Y_input, pred_)
ax.set_xlabel("Actual House price")
ax.set_ylabel("Predicted House Price")
plt.show()


