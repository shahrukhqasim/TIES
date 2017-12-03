import tensorflow as tf
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import configparser as cp
import gzip
import pickle
import os
from tensorflow.contrib.ndlstm.python import lstm2d as lstm2d_lib
import numpy as np
import cv2

slim = tf.contrib.slim


class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        # The file is compressed, so load it using gzip
        f = gzip.open(os.path.join(full_path, '__dump__.pklz'), 'rb')
        doc = pickle.load(f)
        f.close()

        # We don't need to anything with this
        input_tensor = doc.input_tensor

        # Convert left-share to one-hot encoding
        left_class = doc.classes_tensor[:,:,0]
        left_class_one_hot = np.zeros((256,256,2))
        left_class_one_hot[left_class==0, 0] = 1
        left_class_one_hot[left_class==1, 1] = 1

        # Convert top-share to one-hot encoding
        top_class = doc.classes_tensor[:,:,1]
        top_class_one_hot = np.zeros((256,256,2))
        top_class_one_hot[top_class==0, 0] = 1
        top_class_one_hot[top_class==1, 1] = 1

        loss_mask = doc.zone_mask * doc.word_mask

        return input_tensor, left_class_one_hot, top_class_one_hot, loss_mask

class Parser2d:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['zone_segment']['train_data_path']
        self.test_path = config['zone_segment']['test_data_path']
        self.validation_data_path = config['zone_segment']['validation_data_path']
        self.learning_rate = float(config['zone_segment']['learning_rate'])
        self.save_after = int(config['zone_segment']['save_after'])
        self.model_path = config['zone_segment']['model_path']
        self.from_scratch = int(config['zone_segment']['from_scratch']) == 1
        self.batch_size = int(config['zone_segment']['batch_size'])
        self.summary_path = config['zone_segment']['summary_path']

        self.alpha_left = float(config['zone_segment']['alpha_left'])
        self.alpha_top = float(config['zone_segment']['alpha_top'])

        # For usage from other functions
        self.input_placeholder = None
        self.classifier_left_same_placeholder = None
        self.classifier_top_same_placeholder = None
        self.optimizer = None
        self.loss_mask_placeholder = None
        self.loss = None
        self.loss_left = None
        self.loss_top = None

    @staticmethod
    def arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def construct_graphs(self):
        self.input_placeholder = tf.placeholder("float32", shape=[self.batch_size, 256, 256, 308])
        self.classifier_left_same_placeholder = tf.placeholder("float32", shape=[self.batch_size, 256, 256, 2])
        self.classifier_top_same_placeholder = tf.placeholder("float32", shape=[self.batch_size, 256, 256, 2])
        self.loss_mask_placeholder = tf.placeholder("float32", shape=[self.batch_size, 256, 256])

        network = lstm2d_lib.separable_lstm(self.input_placeholder, 100) # (B,256,256,100)

        with slim.arg_scope(Parser2d.arg_scope()):
            output_left_same = slim.conv2d(network, 2, [1, 1], scope='logits_left_same') # (B,256,256,2)
            output_top_same = slim.conv2d(network, 2, [1, 1], scope='logits_top_same') # (B,256,256,2)

        # Apply softmax cross entropy
        loss_left = tf.nn.softmax_cross_entropy_with_logits(labels=self.classifier_left_same_placeholder, logits=output_left_same)
        loss_top = tf.nn.softmax_cross_entropy_with_logits(labels=self.classifier_top_same_placeholder, logits=output_top_same)

        # Mask the loss
        loss_left = tf.multiply(loss_left, self.loss_mask_placeholder)
        loss_top = tf.multiply(loss_top, self.loss_mask_placeholder)

        # Reduce mean (over only the masked indices)
        # Reducing twice in axis 1 will keep the batch dimension (as required)
        num_words = tf.reduce_sum(tf.reduce_sum(self.loss_mask_placeholder, axis=1), axis=1) # (B)
        # Finally, we have to reduce mean across batch dimension
        self.loss_left = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(loss_left, axis=1), axis=1) / num_words)
        self.loss_top = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(loss_top, axis=1), axis=1) / num_words)

        left_predicted_class = tf.argmax(output_left_same, axis=3) #(B,H,W)
        top_predicted_class = tf.argmax(output_left_same, axis=3)

        left_gt_class = tf.argmax(self.classifier_left_same_placeholder, axis=3) #(B,H,W)
        top_gt_class = tf.argmax(self.classifier_top_same_placeholder, axis=3)

        self.accuracy_left = tf.reduce_sum(tf.reduce_sum(
            tf.cast(tf.equal(left_predicted_class, left_gt_class), tf.float32) * self.loss_mask_placeholder, axis=1), axis=1) / num_words # (B)
        self.accuracy_top = tf.reduce_sum(tf.reduce_sum(
            tf.cast(tf.equal(top_predicted_class, top_gt_class), tf.float32) * self.loss_mask_placeholder, axis=1), axis=1) / num_words # (B)

        self.positive_predicted_left = tf.reduce_sum(
            tf.reduce_sum(tf.cast(tf.equal(left_predicted_class, tf.zeros_like(left_predicted_class)), tf.float32) * self.loss_mask_placeholder, axis=1),
            axis=1) / num_words
        self.positive_predicted_top = tf.reduce_sum(
            tf.reduce_sum(tf.cast(tf.equal(top_predicted_class, tf.zeros_like(top_predicted_class)), tf.float32) * self.loss_mask_placeholder, axis=1),
            axis=1) / num_words

        self.num_words = num_words

        # Combine both top and left loss
        self.loss = self.alpha_left * self.loss_left + self.alpha_top * self.loss_top

        summary_loss_complete = tf.summary.scalar('loss_complete', self.loss)
        summary_loss_left = tf.summary.scalar('loss_left', self.loss_left)
        summary_loss_top = tf.summary.scalar('loss_top', self.loss_top)
        summary_accuracy_top = tf.summary.scalar('accuracy_top', tf.reduce_mean(self.accuracy_top))
        summary_accuracy_left = tf.summary.scalar('accuracy_left', tf.reduce_mean(self.accuracy_left))
        summary_predicted_left_zero = tf.summary.scalar('predicted_left_zero', tf.reduce_mean(self.positive_predicted_left))
        summary_predicted_top_zero = tf.summary.scalar('predicted_top_zero', tf.reduce_mean(self.positive_predicted_left))

        self.summaries = tf.summary.merge(
            [summary_loss_complete, summary_loss_left, summary_loss_top, summary_accuracy_top, summary_accuracy_left,
             summary_predicted_left_zero, summary_predicted_top_zero])


        # Optimizer is Adam
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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
        train_set = FolderDataReader(self.train_path, DataLoader())
        train_set.init()
        init = tf.global_variables_initializer()

        print("\n\nNOTE: The cost in the following log will be mean across batch. However, accuracies and positive figures will"
              "be for the first data point in the batch for better debugging.\n\n")

        with tf.Session() as sess:
            if self.from_scratch:
                self.clean_summary_dir()

            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
            sess.run(init)


            if not self.from_scratch:
                self.saver_all.restore(sess, self.model_path)
                with open(self.model_path+'.txt', 'r') as f:
                    iteration = int(f.read())
            else:
                iteration = 0


            while True:
                # Save the model and iteration number to ckpt and txt files respectively
                if iteration % self.save_after == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_all.save(sess, self.model_path)
                    with open(self.model_path+'.txt', 'w') as f:
                        f.write(str(iteration))

                data, epochs, ids = train_set.next_batch(self.batch_size)
                inputs = [data[i][0] for i in range(len(data))]
                left_same = [data[i][1] for i in range(len(data))]
                top_same = [data[i][2] for i in range(len(data))]
                loss_mask = [data[i][3] for i in range(len(data))]

                input_feed = {
                    self.input_placeholder : inputs,
                    self.classifier_left_same_placeholder : left_same,
                    self.classifier_top_same_placeholder : top_same,
                    self.loss_mask_placeholder : loss_mask
                }

                run_ops = [self.optimizer, self.loss, self.accuracy_left, self.accuracy_top,
                           self.positive_predicted_left, self.positive_predicted_top, self.summaries]

                ops_results = sess.run(run_ops, feed_dict=input_feed)

                print("Cost", ops_results[1], "Accuracy Left", ops_results[2][0], "Positive Left",
                      ops_results[4][0], "Accuracy Top", ops_results[3][0], "Positive top", ops_results[5][0])

                summary_writer.add_summary(ops_results[6], iteration)
                iteration += 1




if __name__ == '__main__':
    parser = Parser2d()
    parser.construct_graphs()
    parser.train()