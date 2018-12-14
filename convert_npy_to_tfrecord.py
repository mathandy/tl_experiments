import tensorflow as tf
import numpy as np


# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def npy_to_tfrecords(npy, tfr):
#     writer = tf.python_io.TFRecordWriter(tfr)
#     for i in range(len(train_addrs)):
#         # print how many images are saved every 1000 images
#         if not i % 1000:
#             print 'Train data: {}/{}'.format(i, len(train_addrs))
#             sys.stdout.flush()
#         # Load the image
#         img = load_image(train_addrs[i])
#         label = train_labels[i]
#         # Create a feature
#         feature = {'train/label': _int64_feature(label),
#                    'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#         # Create an example protocol buffer
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#         # Serialize to string and write on the file
#         writer.write(example.SerializeToString())
#
#     writer.close()
#     sys.stdout.flush()
#
# def npy_dir_to_tfrecordsdir(npy, tft):


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})