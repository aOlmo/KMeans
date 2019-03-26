import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)

X = pd.read_csv("breast-cancer-wisconsin.data").iloc[:, 1:-1].values
X[X=='?'] = np.nan
# X = X.astype(int)

n = 2
k = 3
X = X[:n]

def get_centroid_points(closest_centroids, c_dict):
    for i, cc in enumerate(closest_centroids):
        c_dict[cc].append(X[i])

    for i in range(k):
        if (len(c_dict[i]) > 0):
            c_dict[i] = np.stack(c_dict[i], axis=0)

    return c_dict

def assign_centroids(c_dict, centroids):
    dists = np.empty((0, n), int)

    with tf.Session() as sess:
        for c_j in centroids:
            c_j = np.expand_dims(c_j, axis=0)
            dists = np.vstack((dists, sess.run(eucl_dist, feed_dict={centroid: c_j, data_point: X})))

        closest_centroids = np.argmin(dists, axis=0)
        # This function updates c_dictby reference
        get_centroid_points(closest_centroids, c_dict)

if __name__ == '__main__':

    c_dict = {}
    for i in range(k):
        c_dict[i] = []

    # centroids = np.ones((k, 9))
    centroids = np.random.randint(11, size=(k, 9))

    data_point = tf.placeholder("float", [None, 9])
    centroid = tf.placeholder("float", [None, 9])

    eucl_dist = tf.sqrt(tf.reduce_sum(tf.pow((centroid-data_point), 2), reduction_indices=1))

    assign_centroids(c_dict, centroids)
    # TODO: Get new centroids

    print(c_dict)





# Assign closest centroid to each point
# Recalculate centroids from mean
