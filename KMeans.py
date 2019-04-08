import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.datasets.samples_generator import make_blobs


np.random.seed(0)

class KMeans:
    def __init__(self, k, X):
        self.k = k
        self.X = X
        self.cols = self.X.shape[1]
        # By doing this we know that the maximum value
        # that we can obtain for each column is 10
        self.max_val = int(np.amax(X, axis=0)[0]) + 1

        self.potential_function = 0
        self.c_dict = self.init_c_dict()
        self.centroids = np.random.randint(self.max_val, size=(self.k, self.cols))
        self.data_point = tf.placeholder("float", [None, self.cols])

        self.data_point = tf.where(tf.is_nan(self.data_point), tf.zeros_like(self.data_point), self.data_point)
        self.centroid = tf.placeholder("float", [None, self.cols])
        self.eucl_dist = tf.sqrt(tf.reduce_sum(tf.square(self.centroid-self.data_point), reduction_indices=1))

    def init_c_dict(self):
        c_dict = {}
        for i in range(self.k):
            c_dict[i] = np.empty((0, self.cols), float)

        return c_dict

    def get_centroid_points(self, closest_centroids):
        self.c_dict = self.init_c_dict()
        for i, cc in enumerate(closest_centroids):
            self.c_dict[cc] = np.vstack((self.c_dict[cc], self.X[i]))

        for i in range(self.k):
            if (len(self.c_dict[i]) > 0):
                self.c_dict[i] = np.stack(self.c_dict[i], axis=0)


    def assign_centroids(self, sess):
        dists = np.empty((0, self.X.shape[0]), float)

        for c_j in self.centroids:
            c_j = np.expand_dims(c_j, axis=0)
            dists = np.vstack((dists, sess.run(self.eucl_dist, feed_dict={self.centroid: c_j, self.data_point: self.X})))

        closest_centroids = np.argmin(dists, axis=0)
        self.get_centroid_points(closest_centroids)

    def find_largest_cluster(self):
        l_cluster = 0
        l_cluster_sz = 0
        for i, c_point in enumerate(self.c_dict.values()):
            count = c_point.shape[0]
            if (count > l_cluster_sz):
                l_cluster_sz = count
                l_cluster = i

        return l_cluster, l_cluster_sz

    #TODO: Do a shuffle here
    def get_half_of_largest_cluster(self):
        l_c, l_c_sz = self.find_largest_cluster()
        mid = int(l_c_sz/2)
        ret_array = np.copy(self.c_dict[l_c][:mid])
        self.c_dict[l_c] = np.delete(self.c_dict[l_c], range(mid), axis=0)
        return ret_array

    def recalculate_centroids(self):
        new_centroids = []
        empty_cluster = -1
        for i, c_point in enumerate(self.c_dict.values()):
            aux_sum = np.sum(c_point, axis=0)
            count = c_point.shape[0]
            if (count == 0):
                centroid = []
                empty_cluster = i
            else:
                centroid = list(aux_sum / count)

            new_centroids.append(centroid)

        # If there is an empty cluster
        if (empty_cluster > -1):
            # Move half of the points of the largest cluster to this
            self.c_dict[empty_cluster] = self.get_half_of_largest_cluster()
            # And recalculate centroids
            return self.recalculate_centroids()

        if (empty_cluster == -1):
            return np.array(new_centroids)


    def calc_potential_function(self, sess):
        self.potential_function = 0
        for i, X_aux in enumerate(self.c_dict.values()):
            centroid = np.expand_dims(self.centroids[i], axis=0)
            l2_norm = sess.run(self.eucl_dist, feed_dict={self.centroid: centroid, self.data_point: X_aux})
            self.potential_function += np.sum(l2_norm)

    def train(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            converged = False
            count = 0
            while (not converged):
                count += 1

                prev_centroids = self.centroids
                # Assign closest centroid to each data point
                self.assign_centroids(sess)

                # Calculate the new centroids by doing the mean of those assigned
                self.centroids = self.recalculate_centroids()

                # Converges when centroids from one iteration to the next do not vary
                converged = np.allclose(prev_centroids, self.centroids)

                if ((count % 100 == 0) or converged):
                    c_str = ""
                    if (converged):
                        c_str = " Converged at "
                    self.calc_potential_function(sess)
                    print("[+] K: {} -{}Epoch: {}, Loss: {}".format(self.k, c_str, count, self.potential_function))


    def get_potential_function(self):
        return self.potential_function

    def get_dict_and_centroids(self):
        return self.c_dict, self.centroids



def test_with_blobs(k, X, y):
    # Generate 2d classification dataset

    km = KMeans(k, X)
    km.train()
    d, c = km.get_dict_and_centroids()

    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    for c_x, c_y in c:
        plt.plot([c_x], [c_y], 'kx', markersize=5)
    plt.show()


def exercise_4_3():
    p_funcs = []
    Ks = [2, 3, 4, 5, 6, 7, 8]
    for k in Ks:
        km = KMeans(k, X)
        km.train()
        p_funcs.append(km.get_potential_function())

    plt.title("Potential function vs K")
    plt.plot(p_funcs, Ks, 'bo-')
    plt.legend(loc='lower right')
    plt.show()


def exercise_4_2():
    k = 2
    X = np.array([[3,3], [7,9], [9,7], [5,3]])
    y = np.array([0, 1, 1, 0])
    test_with_blobs(k, X, y)


def blob_test():
    k = 2
    n_samples = 200
    X, y = make_blobs(n_samples=n_samples, centers=k, n_features=2)
    test_with_blobs(k, X, y)


if __name__ == '__main__':
    X = pd.read_csv("breast-cancer-wisconsin.data").iloc[:, 1:-1].values
    X[X == '?'] = 0
    X = X.astype(float)

    # Exercise 4.3
    exercise_4_3()

    # Exercise 4.2
    exercise_4_2()

    # More testing
    blob_test()


