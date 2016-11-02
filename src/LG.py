mport gzip
import os
import sys
import numpy as np
import cPickle
 
import tensorflow as tf
 
 
class LogisticRegression(object):
    def __init__(self):
        self.X = tf.placeholder("float", [None, 784])
        self.Y = tf.placeholder("float", [None, 10])
 
        self.W = tf.Variable(tf.random_normal([28 * 28, 10], stddev=0.01))
        self.b = tf.Variable(tf.zeros([10, ]))
 
        self.model = self.create_model(self.X, self.W, self.b)
 
        # logistic and cal error
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.Y))
 
        # gradient descent method to minimize error
        self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)
        # calculate the max pos each row
        self.predict = tf.argmax(self.model, 1)
 
 
    def create_model(self, X, w, b):
        # wx + b
        return tf.add(tf.matmul(X, w), b)
 
 
    def load_data(self):
        
		"""
		f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
		"""
        return train_set, valid_set, test_set
 
 
    def dense_to_one_hot(self, labels_dense, num_classes=10):
        # ont hot copy from https://github.com/nlintz/TensorFlow-Tutorials
        # also can use sklearn preprocessing OneHotEncoder()
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
 
 
    def run(self):
        train_set, valid_set, test_set = self.load_data()
        train_X, train_Y = train_set
        test_X, test_Y = test_set
        train_Y = self.dense_to_one_hot(train_Y)
        test_Y = self.dense_to_one_hot(test_Y)
 
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
 
        for i in range(100):
            for start, end in zip(range(0, len(train_X), 128), range(128, len(train_X), 128)):
                sess.run(self.train, feed_dict={self.X: train_X[start:end], self.Y: train_Y[start:end]})
            print i, np.mean(np.argmax(test_Y, axis=1) == sess.run(self.predict, feed_dict={self.X: test_X, self.Y: test_Y}))
 
        sess.close()
 
 
if __name__ == '__main__':
    lr_model = LogisticRegression()
    lr_model.run()
