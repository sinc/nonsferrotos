﻿"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.
Parag K. Mital, Jan 2016
"""
import sys
import tensorflow as tf
import numpy as np
import math
from tf_utils import corrupt, lrelu

sys.path.append('..\\..\\')
import nonsferrotos.models.ferros_dataset as input_data

# %%
def autoencoder(input_shape=[None, 900],
                n_filters=[1, 200, 150, 100],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    out = tf.placeholder(tf.float32, input_shape, name='out')
    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    out_tensor = tf.reshape(out, [-1, x_dim, x_dim, n_filters[0]])

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - out_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'out': out}


# %%
def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    #import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt


    # %%
    # load MNIST as before
    data_set = input_data.make_data_set()
    #data_set.train.save_dataset('temp.npz')
    #mean_img = np.mean(data_set.train.images, axis=0)
    print('Images size is:' + str(data_set.train.image_size))
    ae = autoencoder(input_shape = [None, data_set.train.image_size])

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(ae['cost'])
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 5000
    for epoch_i in range(n_epochs):
        for batch_i in range(data_set.train.num_examples // batch_size):
            batch_xs, batch_ys = data_set.train.next_batch(batch_size)
            train_in = np.array([img for img in batch_xs])
            train_out = np.array([lbl for lbl in batch_ys])
            sess.run(optimizer, feed_dict={ae['x']: train_in, ae['out']:train_out})
        error = sess.run(ae['cost'], feed_dict={ae['x']: train_in, ae['out']:train_out})
        print(epoch_i,error)
        if (error < 3000):
            break

    # %%
    # Plot example reconstructions
    n_examples = 10
    test_xs, test_ys = data_set.test.next_batch(n_examples)
    test_xs_norm = np.array([img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (50, 50)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (2500,)),
                (50, 50)))
    #fig.show()
    plt.show()
    #plt.waitforbuttonpress()


# %%
if __name__ == '__main__':
    test_mnist()