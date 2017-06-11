from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()


def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]), name="W")
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return act


def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in, channels_out]), name="W")
        b = tf.Variable(tf.zeros([channels_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act


x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(x_image, 1, 32, name="conv1")
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2 = conv_layer(pool1, 32, 64, name="conv2")
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
logits = fc_layer(fc1, 1024, 10, "fc2")

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter('logs/mnist_demo/2')
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y: batch[1]})
