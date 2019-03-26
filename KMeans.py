import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv("breast-cancer-wisconsin.data")
to_np = df.iloc[:, 1:-1]
X = to_np.values
X[X=='?'] = np.nan
# X = X.astype(int)

k = 3
# Initialize centroids
c_j = np.expand_dims([1]*X.shape[1], axis=0)
# c_j = np.tile(c_j, k)
# c_j = np.expand_dims(c_j.reshape([-1, 9]), axis=1)

x_i = X[:2]
# x_i = np.tile(x_i, k)
# x_i = np.expand_dims(np.reshape(x_i, [-1, 9]), axis=0)

np.random.seed(0)
# centroids = np.ones((k, 9))
centroids = np.random.randint(11, size=(k, 9))

data_point = tf.placeholder("float", [None, 9])
centroid = tf.placeholder("float", [None, 9])

eucl_dist = tf.sqrt(tf.reduce_sum(tf.pow((centroid-data_point), 2), reduction_indices=1))

with tf.Session() as sess:
    for c_j in centroids:
        c_j = np.expand_dims(c_j, axis=0)
        print(sess.run(eucl_dist, feed_dict={centroid: c_j, data_point: x_i}))



# Assign closest centroid to each point
# Recalculate centroids from mean
