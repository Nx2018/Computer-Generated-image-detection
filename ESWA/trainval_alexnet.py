#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from datetime import datetime
import cv2
import os
import time
from alexnet_model import alexnet
from datagenerator import ImageDataGenerator

funcs = ['fine_tuning', 'predict']
which_func = funcs[1]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class alexnet_train_val(object):
    def __init__(self):
        self.PRE_MODEL = './bvlc_alexnet.npy'
        self._start_end_time = [0,0]
        
    def fine_tuning(self, train_list, test_list, mean, snapshot, filewriter_path):
        # Learning params
        learning_rate = 0.0005
        num_epochs = 151
        batch_size = 64

        # Network params
        in_img_size = (227, 227) #(height, width)
        dropout_rate = 1
        num_classes = 2
        train_layers = ['fc7', 'fc8']

        # How often we want to write the tf.summary data to disk
        display_step = 30
        
        x = tf.placeholder(tf.float32, [batch_size, in_img_size[0], in_img_size[1], 3])
        y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)
        
        # Initialize model
        model = alexnet(x, keep_prob, num_classes, train_layers, in_size=in_img_size)
        #link variable to model output
        score = model.fc8
        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))
        # Train op
        
            # Get gradients of all trainable variables
            gradients = tf.gradients(loss, var_list)
            gradients = list(zip(gradients, var_list))
            '''
            # Create optimizer and apply gradient descent to the trainable variables
            learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=tf.Variable(0, trainable=False),
                                           decay_steps=10,decay_rate=0.9)
            '''
            optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
            train_op = optimizer.minimize(loss)

        # Add gradients to summary
        for gradient, var in gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)
        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)
        # Add the loss to summary
        tf.summary.scalar('cross_entropy', loss)
        
        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()
        # Initialize the FileWriter
        writer = tf.summary.FileWriter(filewriter_path)
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()
        # Initalize the data generator seperately for the training and validation set
        train_generator = ImageDataGenerator(train_list, horizontal_flip = True, shuffle = False, 
                                             mean=mean, scale_size=in_img_size, nb_classes=num_classes)
        val_generator = ImageDataGenerator(test_list, shuffle = False, 
                                             mean=mean, scale_size=in_img_size, nb_classes=num_classes)
        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
        
        # Start Tensorflow session
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)
            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)
            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))
            # Loop over number of epochs
            for epoch in range(num_epochs):
                print("{} Epoch number: {}/{}".format(datetime.now(), epoch+1, num_epochs))
                step = 1
                while step < train_batches_per_epoch:
                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(batch_size)
                    # And run the training op
                    sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})
                    # Generate summary with the current batch of data and write to file
                    if step%display_step == 0:
                        s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                        writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_generator.next_batch(batch_size)
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

                # Reset the file pointer of the image data generator
                val_generator.reset_pointer()
                train_generator.reset_pointer()
                print("{} Saving checkpoint of model...".format(datetime.now()))

                #save checkpoint of the model
                if epoch % display_step == 0:
                    checkpoint_name = os.path.join(snapshot, 'model_epoch'+str(epoch)+'.ckpt')
                    save_path = saver.save(sess, checkpoint_name)
                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                
                
    def predict_batch(self, val_list, mean, weight_file, result_file):
        in_img_size = (227, 227) #(height, width)
        dropout_rate = 0.5
        num_classes = 2
        train_layers = []
        
        x = tf.placeholder(tf.float32, [1, in_img_size[0], in_img_size[1], 3])
        y = tf.placeholder(tf.float32, [None, num_classes])
        
        model = alexnet(x, 1., num_classes, train_layers, in_size=in_img_size, weights_path=weight_file)
        score = model.fc8
        softmax = tf.nn.softmax(score)
        
        val_generator = ImageDataGenerator(val_list, horizontal_flip = False, shuffle = False, 
                                             mean=mean, scale_size=in_img_size, nb_classes=num_classes)
        
        precision = np.zeros((num_classes+1, num_classes), dtype=np.float)
        total_presion = 0.
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, weight_file)
            
            self._start_end_time[0] = time.clock()
            for index in range(val_generator.data_size):
                print 'handing %d / %d ...\r'%(index+1, val_generator.data_size),
                
                img_ = val_generator.images[index]
                label = val_generator.labels[index]
                img = cv2.imread(img_)
                img = cv2.resize(img, (val_generator.scale_size[1], val_generator.scale_size[0]))
                img = img.reshape(1, val_generator.scale_size[0], val_generator.scale_size[1], 3)
                img = img.astype(np.float32)
                
                probs = sess.run(softmax, feed_dict={x: img})
                guess = np.argmax(probs)
                if guess == label:
                    precision[guess][guess] += 1
                    total_presion += 1
                else:
                    precision[guess][int(val_generator.labels[index])] += 1
            self._start_end_time[1] = time.clock()
            
            for i in range(num_classes):
                for j in range(num_classes):
                    precision[num_classes][i] += precision[j][i]
            for i in range(num_classes):
                for j in range(num_classes):
                    precision[i][j] /= precision[num_classes][j]
            total_presion /= val_generator.data_size
        
            slaped = (self._start_end_time[1] - self._start_end_time[0]) / val_generator.data_size
            
            file = open(result_file, 'w')
            file.write('model: ' + weight_file + '\n')
            print '\n#####################################################################'
            file.writelines(['################################################################\n'])
            text_ = ''
            for i in range(num_classes):
                print '        %d'%i,
                text_ += '        %d'%i
            print '\n'
            file.write(text_ + '\n')
            for i in range(num_classes):
                print '  %d'%i,
                file.write('  ' + str(i))
                for j in range(num_classes):
                    str_preci = '    %.2f'%precision[i][j]
                    print '  %.2f  '%precision[i][j],
                    file.write(str_preci)
                print '\n'
                file.write('\n')
            print '\ntotal precision: %.2f'%total_presion
            print 'average speed: %.4f / image'%slaped
            str_preci = 'total precision: %.2f'%total_presion
            file.writelines(['\n' + str_preci + '\n'])
            str_slaped = 'average speed: %.4f s / image'%slaped
            file.write(str_slaped + '\n')
            file.close()
                
                
def main():
    base_path = './4850lr0.0005_loss/'
    data_path = './'
    
    alex = alexnet_train_val()
    if which_func == funcs[0]:
        # fine tuning
        train_list = data_path + 'train4850.txt'
        test_list = data_path + 'test4850.txt'
        snapshot = base_path
        filewriter = base_path + 'train_log'
        mean_value = np.array([])
        
        alex.fine_tuning(train_list, test_list, mean_value, snapshot, filewriter)
    elif which_func == funcs[1]:
        # validate
        #val_list = data_path + 'val.txt'
        val_list = data_path + '4850gauss0_50.txt'
        mean_value = np.array([])
        result_file = base_path + '4850gauss0_50.txt'
        checkpoint_file = base_path + 'model_epoch150.ckpt'
        
        alex.predict_batch(val_list, mean_value, checkpoint_file, result_file)
        
        
if __name__ == '__main__':
    main()