import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)

X = pd.read_csv("breast-cancer-wisconsin.data").iloc[:, 1:-1].values
X[X=='?'] = np.nan
X = X.astype(float)

n = 100
k = 2
X = X[:n]

def get_centroid_points(closest_centroids, c_dict):
    for i, cc in enumerate(closest_centroids):
        c_dict[cc] = np.vstack((c_dict[cc], X[i]))

    for i in range(k):
        if (len(c_dict[i]) > 0):
            c_dict[i] = np.stack(c_dict[i], axis=0)


def assign_centroids(c_dict, centroids, eucl_dist, sess):
    dists = np.empty((0, n), float)


    for c_j in centroids:
        c_j = np.expand_dims(c_j, axis=0)
        dists = np.vstack((dists, sess.run(eucl_dist, feed_dict={centroid: c_j, data_point: X})))

    closest_centroids = np.argmin(dists, axis=0)
    # This function updates c_dictby reference
    get_centroid_points(closest_centroids, c_dict)


def recalculate_centroids(c_dict):
    new_centroids = []
    for i, c_point in enumerate(c_dict.values()):
        new_centroids.append(list(np.sum(c_point, axis=0)/(c_point.shape[0])))

    return np.array(new_centroids)

if __name__ == '__main__':

    c_dict = {}
    for i in range(k):
        c_dict[i] = np.empty((0, 9), float)

    centroids = np.random.randint(11, size=(k, 9))
    data_point = tf.placeholder("float", [None, 9])
    centroid = tf.placeholder("float", [None, 9])

    eucl_dist = tf.sqrt(tf.reduce_sum(tf.pow((centroid-data_point), 2), reduction_indices=1))
    with tf.Session() as sess:
        converged = False
        while (not converged):
            prev_centroids = centroids
            # Assign closest centroid to each data point
            assign_centroids(c_dict, centroids, eucl_dist, sess)

            # Calculate the new centroids by doing the mean of those assigned
            centroids = recalculate_centroids(c_dict)

            # Converges when centroids from one iteration to the next do not vary: converged = prev_iter == curr_iter
            converged = np.allclose(prev_centroids, centroids, equal_nan=True)



# Assign closest centroid to each point
# Recalculate centroids from mean
