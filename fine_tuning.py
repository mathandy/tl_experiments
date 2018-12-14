#!/usr/bin/env python
""" 

Some code taken from:
https://github.com/tensorflow/hub/blob/master/examples/colab/image_feature_vector.ipynb
"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


# User parameters
LEARNING_RATE = 0.01
NUM_CLASSES = 10
TFHUB_MODULE = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
DEBUG = True


if DEBUG:
    tf.enable_eager_execution()

def load_model():

    tf.reset_default_graph()
    pretrained_model = hub.Module(TFHUB_MODULE)

    logits = tf.layers.dense(inputs=features, units=NUM_CLASSES, activation=None)
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss=cross_entropy_mean)

    # output probabilities
    probabilities = tf.nn.softmax(logits)

    # argmax/classification step
    prediction = tf.argmax(probabilities, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))

    # define useful metrics
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
      for i in range(NUM_TRAIN_STEPS):
        # Get a random batch of training examples.
        train_batch = get_batch(batch_size=TRAIN_BATCH_SIZE)
        batch_images, batch_labels = get_images_and_labels(train_batch)
        # Run the train_op to train the model.
        train_loss, _, train_accuracy = sess.run(
            [cross_entropy_mean, train_op, accuracy],
            feed_dict={encoded_images: batch_images, labels: batch_labels})
        is_final_step = (i == (NUM_TRAIN_STEPS - 1))
        if i % EVAL_EVERY == 0 or is_final_step:
          # Get a batch of test examples.
          test_batch = get_batch(batch_size=None, test=True)
          batch_images, batch_labels = get_images_and_labels(test_batch)
          # Evaluate how well our model performs on the test set.
          test_loss, test_accuracy, test_prediction, correct_predicate = sess.run(
            [cross_entropy_mean, accuracy, prediction, correct_prediction],
            feed_dict={encoded_images: batch_images, labels: batch_labels})
          print('Test accuracy at step %s: %.2f%%' % (i, (test_accuracy * 100)))
if __name__ == '__main__':
    image_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once('images/apple/*.jpg'))
    decode_and_resize_image(image_filename_queue)
