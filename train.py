from core.model import ShowAndTellModel
from core.utils import *
from core.bleu import evaluate

import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import os 
import cPickle as pickle
from scipy import ndimage


def main(unused_argv):
    # load preprocessed MSCOCO data
    data = load_coco_data(data_path='/home/yifan/PythonProjects/im2txt-att/data', split='train')
    word_to_idx = data['word_to_idx']
    #vocab_size = len(word_to_idx)
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='/home/yifan/PythonProjects/im2txt-att/data', split='val')
    
    features = data['features']
    captions = data['captions']
    image_idxs = data['image_idxs']
    val_features = val_data['features']
    
    # train/val settings
    n_epochs = 1
    batch_size = 128
    num_examples_per_epoch = data['captions'].shape[0]
    num_batches_per_epoch = (num_examples_per_epoch / batch_size)
    n_iters_per_epoch = int(np.ceil(float(num_examples_per_epoch)/batch_size))
    
    learning_rate = tf.constant(0.01)
    num_epochs_per_decay = 4.0
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    learning_rate_decay_factor = 0.5
    clip_gradients = 5.0
    optimizer = "SGD"    
    
    # basic settings
    restore_last_model=True
    save_every = 1
    
    # path
    model_path='/home/yifan/PythonProjects/im2txt-att/model/lstm/'
    log_path='/home/yifan/PythonProjects/im2txt-att/log/'
    
    
    print("The number of epoch: %d" %n_epochs)
    print("Data size: %d" %num_examples_per_epoch)
    print("Batch size: %d" %batch_size)
    print("Iterations per epoch: %d" %n_iters_per_epoch)
    
    # build show_and_tell model
    model = ShowAndTellModel(word_to_idx, mode='train', dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, alpha_c=1.0)
    
    # build graphs for training model and sampling captions
    loss = model.build_model()
    tf.get_variable_scope().reuse_variables()    
    
    def _learning_rate_decay_fn(learning_rate, global_step):
      return tf.train.exponential_decay(
          learning_rate,
          global_step,
          decay_steps=decay_steps,
          decay_rate=learning_rate_decay_factor,
          staircase=True)

    learning_rate_decay_fn = _learning_rate_decay_fn
    
    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)
       
    # summary op 
    tf.summary.scalar('batch_loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    summary_op = tf.summary.merge_all() 
        
    # Run training.
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=20)
        # restore the latest trained model
        if restore_last_model:
            print('Start restoring pretrained model..')
            saver.restore(sess, os.path.join(model_path, 'model.ckpt'))
            global_step = tf.train.global_step(sess, model.global_step.name)
            print("global step restored: ", global_step)

        for epoch in range(n_epochs):
            rand_idxs = np.random.permutation(num_examples_per_epoch)
            captions = captions[rand_idxs]
            image_idxs = image_idxs[rand_idxs]
            for i in range(n_iters_per_epoch):
                captions_batch = captions[i*batch_size:(i+1)*batch_size]
                image_idxs_batch = image_idxs[i*batch_size:(i+1)*batch_size]
                features_batch = features[image_idxs_batch]
                feed_dict = {model.features: features_batch, model.captions: captions_batch}
                _, l = sess.run([train_op, loss], feed_dict)
                print ("Train loss at epoch %d & iteration %d: %.5f" %(epoch+1, i+1, l))
                # write summary for tensorboard visualization
                if i % 10 == 0:
                    global_step = tf.train.global_step(sess, model.global_step.name)
                    print("summary global step: ", global_step)
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, global_step)

            # save model's parameters
            if (epoch+1) % save_every == 0:
                saver.save(sess, os.path.join(model_path, 'model.ckpt'))
                print("model-%s saved." %(epoch+1))
                
if __name__ == "__main__":
  tf.app.run()