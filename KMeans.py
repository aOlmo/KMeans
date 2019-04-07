import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)

X = pd.read_csv("breast-cancer-wisconsin.data").iloc[:, 1:-1].values
X[X=='?'] = 0
X = X.astype(float)

cols = X.shape[1]

# By doing this we know that the maximum value
# that we can obtain for each column is 10
max_val = int(np.amax(X, axis=0)[0])+1

n = 10
k = 4
X = X[:n]

class KMeans:
    def __init__(self, k, X):
        self.X = X
        self.potential_function = 0
        self.c_dict = self.init_c_dict()
        self.centroids = np.random.randint(max_val, size=(k, cols))
        self.data_point = tf.placeholder("float", [None, cols])

        self.data_point = tf.where(tf.is_nan(self.data_point), tf.zeros_like(self.data_point), self.data_point)
        self.centroid = tf.placeholder("float", [None, cols])
        self.eucl_dist = tf.sqrt(tf.reduce_sum(tf.square(self.centroid-self.data_point), reduction_indices=1))

    def init_c_dict(self):
        c_dict = {}
        for i in range(k):
            c_dict[i] = np.empty((0, cols), float)

        return c_dict

    def get_centroid_points(self, closest_centroids):
        self.c_dict = self.init_c_dict()
        for i, cc in enumerate(closest_centroids):
            self.c_dict[cc] = np.vstack((self.c_dict[cc], X[i]))

        for i in range(k):
            if (len(self.c_dict[i]) > 0):
                self.c_dict[i] = np.stack(self.c_dict[i], axis=0)


    def assign_centroids(self, sess):
        dists = np.empty((0, X.shape[0]), float)

        for c_j in self.centroids:
            c_j = np.expand_dims(c_j, axis=0)
            dists = np.vstack((dists, sess.run(self.eucl_dist, feed_dict={self.centroid: c_j, self.data_point: X})))

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

    def get_half_of_largest_cluster(self):
        l_c, l_c_sz = self.find_largest_cluster()
        # Get the middlepoint
        mid = int(l_c_sz/2)
        #TODO: Do a shuffle here
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
            self.recalculate_centroids()

        print(new_centroids)

        return np.array(new_centroids)

    #TODO: Calculate potential function. Why does not get smaller?
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
                converged = np.allclose(prev_centroids, self.centroids, equal_nan=True, rtol=0.00001)

                if ((count % 100 == 0) or converged):
                    self.calc_potential_function(sess)
                    print("[+] Epoch: {}, Loss: {}".format(count, self.potential_function))

            print("[+]: CONVERGED")

    def get_potential_function(self):
        return self.potential_function

    def get_dict_and_centroids(self):
        return self.c_dict, self.centroids

if __name__ == '__main__':
    km = KMeans(k, X)
    km.train()
    d, c = km.get_dict_and_centroids()
