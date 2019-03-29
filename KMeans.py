import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)

X = pd.read_csv("breast-cancer-wisconsin.data").iloc[:, 1:-1].values
X[X=='?'] = np.nan
X = X.astype(float)

n = 100
k = 9
X = X[:n]

class KMeans:
    def __init__(self, k, X):
        self.X = X
        self.c_dict = self.init_c_dict()
        self.centroids = np.random.randint(11, size=(k, 9))
        self.data_point = tf.placeholder("float", [None, 9])
        self.centroid = tf.placeholder("float", [None, 9])
        self.eucl_dist = tf.sqrt(tf.reduce_sum(tf.pow((self.centroid-self.data_point), 2), reduction_indices=1))

    def init_c_dict(self):
        c_dict = {}
        for i in range(k):
            c_dict[i] = np.empty((0, 9), float)

        return c_dict

    def get_centroid_points(self, closest_centroids):
        for i, cc in enumerate(closest_centroids):
            self.c_dict[cc] = np.vstack((self.c_dict[cc], X[i]))

        for i in range(k):
            if (len(self.c_dict[i]) > 0):
                self.c_dict[i] = np.stack(self.c_dict[i], axis=0)


    def assign_centroids(self, c_dict, centroids, eucl_dist, sess):
        dists = np.empty((0, n), float)

        for c_j in centroids:
            c_j = np.expand_dims(c_j, axis=0)
            dists = np.vstack((dists, sess.run(eucl_dist, feed_dict={self.centroid: c_j, self.data_point: X})))

        closest_centroids = np.argmin(dists, axis=0)
        self.get_centroid_points(closest_centroids)


    def recalculate_centroids(self, c_dict):
        new_centroids = []
        for i, c_point in enumerate(self.c_dict.values()):
            new_centroids.append(list(np.sum(c_point, axis=0)/(c_point.shape[0])))

        return np.array(new_centroids)


    def train(self):
        with tf.Session() as sess:
            converged = False
            while (not converged):
                prev_centroids = self.centroids
                # Assign closest centroid to each data point
                self.assign_centroids(self.c_dict, self.centroids, self.eucl_dist, sess)

                # Calculate the new centroids by doing the mean of those assigned
                self.centroids = self.recalculate_centroids(self.c_dict)

                # Converges when centroids from one iteration to the next do not vary
                converged = np.allclose(prev_centroids, self.centroids, equal_nan=True)

