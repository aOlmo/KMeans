import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)

df = pd.read_csv("breast-cancer-wisconsin.data")
to_np = df.iloc[:, 1:-1]
X = to_np.values
X[X=='?'] = np.nan
# X = X.astype(int)

n = 3
k = 3
# Initialize centroids
X = X[:n]

# centroids = np.ones((k, 9))
centroids = np.random.randint(11, size=(k, 9))

data_point = tf.placeholder("float", [None, 9])
centroid = tf.placeholder("float", [None, 9])

eucl_dist = tf.sqrt(tf.reduce_sum(tf.pow((centroid-data_point), 2), reduction_indices=1))

c_dict = {}
for i in range(k):
    c_dict[i] = []

dists = np.empty((0, n), int)
with tf.Session() as sess:
    for c_j in centroids:
        c_j = np.expand_dims(c_j, axis=0)
        dists = np.vstack((dists, sess.run(eucl_dist, feed_dict={centroid: c_j, data_point: X})))

    closest_centroids = np.argmin(dists, axis=0)

    # get_centroid_points()
    for i, cc in enumerate(closest_centroids):
        c_dict[cc].append(X[i])

    for i in range(k):
        if (len(c_dict[i]) > 0):
            c_dict[i] = np.stack(c_dict[i], axis=0)

    print(c_dict)




# Assign closest centroid to each point
# Recalculate centroids from mean
