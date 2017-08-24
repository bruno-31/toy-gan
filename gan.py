import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.misc
import shutil
import os
import conv_model

max_iter = 20000
batch_size = 256
output_path = 'output'
log_path = 'log'
freq_print = 20
freq_save = 1000
KEEP_RATE = 0.7
IM_HEIGHT = 28
IM_WIDTH = 28
IM_SIZE = IM_HEIGHT * IM_WIDTH


def store_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    # display function
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], IM_HEIGHT, IM_WIDTH)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
        scipy.misc.imsave(fname, img_grid)


def train_gan():
    # collecting data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # placeholders
    x_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, IM_SIZE], name="x_data")
    keep_pl = tf.placeholder(dtype=tf.float32, name="dropout_keep_rate")

    # build model
    with tf.variable_scope("generator_model"):
        x_generated = conv_model.build_generator(batch_size)

    with tf.variable_scope("discriminator_model") as scope:  # we use only one model for discriminator with 2 inputs
        dis_gen = conv_model.build_discriminator(x_generated, keep_pl)
        scope.reuse_variables()
        dis_data = conv_model.build_discriminator(x_data, keep_pl)

    with tf.name_scope('generator_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen, labels=tf.ones_like(dis_gen)))
        dis_logits_on_generated = tf.reduce_mean(tf.sigmoid(dis_gen))
    with tf.name_scope('discriminator_loss'):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_data, labels=tf.fill([batch_size,1],0.9)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen, labels=tf.zeros_like(dis_gen)))
        d_loss = d_loss_fake + d_loss_real
        dis_logits_on_real = tf.reduce_mean(tf.sigmoid(dis_data))

    optimzer = tf.train.AdamOptimizer(0.0001)
    # collecting 2 list of training variables corresponding to discriminator and generator
    tvars = tf.trainable_variables()  # return list trainable variables
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    for var in d_vars:  # display trainable vars for sanity check
        print(var.name)
    for var in g_vars:
        print(var.name)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_trainer = optimzer.minimize(g_loss, var_list=g_vars, name='generator_trainer')

    d_trainer = optimzer.minimize(d_loss, var_list=d_vars, name='discriminator_trainer')

    # summary
    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_real_loss', d_loss_real)
    tf.summary.scalar('Discriminator_fake_loss', d_loss_fake)
    tf.summary.scalar('Discriminator_total_loss', d_loss)
    tf.summary.scalar('logits_discriminator_on_generated', dis_logits_on_generated)
    tf.summary.scalar('logits_discriminator_on_real', dis_logits_on_real)

    tf.summary.image('Generated_images', x_generated, 10)  # add 10 generated images to summary
    x_data_reshaped = tf.reshape(x_data, shape=[-1, 28, 28, 1])
    tf.summary.image('data_images', x_data_reshaped, 10)
    merged = tf.summary.merge_all()

    # train
    with tf.Session() as sess:
        # write tensorflow summary for monitoring on tensorboard
        writer = tf.summary.FileWriter(log_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(max_iter):
            x_batch, _ = mnist.train.next_batch(batch_size)
            x_batch = 2 * x_batch.astype(np.float32) - 1  # set image dynamic to [-1 1]

            sess.run(d_trainer, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
            sess.run(g_trainer, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})

            if i % 1 == 0:
                print("step %d" % (i))

            if i % freq_print == 0:
                summary = sess.run(merged, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
                writer.add_summary(summary, i)

            if i % freq_save == 0:
                sample_images = sess.run(x_generated, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
                store_result(sample_images, os.path.join(output_path, "sample%s.jpg" % i))
                #   saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

        im_g_sample = sess.run(x_generated)
        store_result(im_g_sample, os.path.join(output_path, "final_samples.jpg"))


if __name__ == '__main__':
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.mkdir(log_path)
    print("start training")
    train_gan()
