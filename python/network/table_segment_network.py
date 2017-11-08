import tensorflow as tf
from .silknet import LoadInterface
from .silknet.FolderDataReader import FolderDataReader
from interface import implements
import cv2
import os
import numpy as np
import math


slim = tf.contrib.slim


class DataLoader(implements(LoadInterface)):
    def __init__(self, down_pool_by):
        self.down_pool_by = down_pool_by

    def load_datum(self, full_path):
        image_full = cv2.imread(os.path.join(full_path, 'image.png'), 1)
        image_height, image_width, _  = np.shape(image_full)
        scale_factor = (1000/max(image_height, image_width))
        image_full = cv2.resize(image_full, None, fx=scale_factor, fy=scale_factor)
        image_height_reduced, image_width_reduced, _  = np.shape(image_full)
        tables_segmentation = cv2.imread(os.path.join(full_path, 'tables.png'), 0)
        tables_segmentation = cv2.resize(tables_segmentation, (int(image_width_reduced / self.down_pool_by), int(image_height_reduced / self.down_pool_by)))
        height, width = np.shape(tables_segmentation)
        tables_segmentation2 = np.zeros((height, width, 2))
        tables_segmentation2[:, :, 1] = tables_segmentation / 255
        tables_segmentation2[:, :, 0] = 1 - tables_segmentation2[:, :, 1]

        datum = dict()
        datum['image'] = image_full
        datum['segmentation'] = tables_segmentation2

        return datum



class TableSegmentNetwork:
    def __init__(self):
        self.learning_rate = 0.0001
        self.data_path = '/home/srq/Datasets/tables/unlv-for-segment/train'
        self.loss = None
        self.logits = None
        self.optimizer = None
        self.accuracy = None
        self.down_pool_by = None
        self.from_scratch = True
        self.summary_path = 'summary'
        self.save_after_iterations = 500
        self.model_path = 'models/model1.ckpt'

    @staticmethod
    def arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def construct_graphs(self):
        self.images_placeholder = x = tf.placeholder("float32", shape=[1, None, None, 3])
        self.segmentations_placeholder = y = tf.placeholder("float32", shape=[1, None, None, 2])

        with slim.arg_scope(TableSegmentNetwork.arg_scope()):
            net = slim.conv2d(x, 24, [5, 5], scope='1c')
            net = slim.conv2d(net, 24, [5, 5], scope='2c')
            net = slim.max_pool2d(net, [2, 2], scope='3p')
            net = slim.conv2d(net, 48, [3, 3], scope='4c')
            net = slim.conv2d(net, 24, [3, 3], scope='5c')
            net = slim.max_pool2d(net, [2, 2], scope='6p')
            net = slim.conv2d(net, 24, [3, 3], scope='7p')
            net = slim.conv2d(net, 2, [1, 1], scope='8p')
            logits = tf.nn.softmax(net)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y, axis=3), tf.argmax(net, axis=3))))
            train_loss = tf.summary.scalar('train_loss', loss)
            train_accuracy = tf.summary.scalar('train_accuracy', accuracy)
            self.train_summary = tf.summary.merge([train_loss, train_accuracy])

        self.loss = loss
        self.logits = logits
        self.optimizer = optimizer
        self.down_pool_by = 4
        self.accuracy = accuracy
        self.saver_all = tf.train.Saver()


    def clean_summary_dir(self):
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def train(self):
        dataset = FolderDataReader(self.data_path, DataLoader(self.down_pool_by))
        dataset.init()
        init = tf.global_variables_initializer()

        if self.from_scratch:
            self.clean_summary_dir()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
            sess.run(init)
            if not self.from_scratch:
                self.saver_all.restore(sess, self.model_path)
                with open(self.model_path+'.txt', 'r') as f:
                    iteration = int(f.read())
            else:
                iteration = 0
            while True:
                datum, epoch, id = dataset.next_element()
                x = datum['image']
                y = datum['segmentation']
                c1, a1, train_summary, o1 = sess.run([self.loss, self.accuracy, self.train_summary, self.optimizer],
                                      feed_dict={self.images_placeholder: [x],
                                                 self.segmentations_placeholder: [y]})
                iteration += 1
                print("\tIteration", iteration, "Cost", c1, "Accuracy", a1)
                summary_writer.add_summary(train_summary, iteration)

                if iteration % self.save_after_iterations == 0:
                    self.saver_all.save(sess, self.model_path)
                    print("Saving model")
                    with open(self.model_path+'.txt', 'w') as f:
                        f.write(str(iteration))


