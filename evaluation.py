import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core.model import ShowAndTellModel
from core.utils import *
from core.bleu import evaluate
from scipy import ndimage
import skimage.transform
import numpy as np


def main(unused_argv):
    #matplotlib inline
    plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    data = load_coco_data(data_path='/home/yifan/PythonProjects/im2txt-att/data', split='val')
    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
        
    # val settings
    split='train'
    batch_size = 128
        
    # basic settings
    test_path='/home/yifan/PythonProjects/im2txt-att/model/lstm/model-test'
    model_path='/home/yifan/PythonProjects/im2txt-att/model/lstm/model.ckpt'
    
    model = ShowAndTellModel(word_to_idx, mode='eval', dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, alpha_c=1.0)
    
    # test
    features = data['features']

    # build a graph to sample captions
    alphas, betas, sampled_captions = model.build_model(max_len=20) 
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        features_batch, image_files = sample_coco_minibatch(data, batch_size)
        feed_dict = { model.features: features_batch }
        alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)
        decoded = decode_captions(sam_cap, model.idx_to_word)

        # attention visualization
        for n in range(10):
            print("Sampled Caption: %s" %decoded[n])

            # Plot original image
            img = ndimage.imread(image_files[n])
            plt.subplot(4, 5, 1)
            plt.imshow(img)
            plt.axis('off')

            # Plot images with attention weights 
            words = decoded[n].split(" ")
            for t in range(len(words)):
                if t > 18:
                    break
                plt.subplot(4, 5, t+2)
                plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                plt.imshow(img)
                alp_curr = alps[n,t,:].reshape(14,14)
                alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                plt.imshow(alp_img, alpha=0.85)
                plt.axis('off')
            plt.show()
        
        # print out BLEU scores and file write
        all_sam_cap = np.ndarray((features.shape[0], 20))
        num_iter = int(np.ceil(float(features.shape[0]) / batch_size))
        for i in range(num_iter):
            features_batch = features[i*batch_size:(i+1)*batch_size]
            feed_dict = { model.features: features_batch }
            all_sam_cap[i*batch_size:(i+1)*batch_size] = sess.run(sampled_captions, feed_dict)  
        all_decoded = decode_captions(all_sam_cap, model.idx_to_word)
        save_pickle(all_decoded, "/home/yifan/PythonProjects/im2txt-att/data/%s/%s.candidate.captions.pkl" %(split,split))
        scores = evaluate(data_path='/home/yifan/PythonProjects/im2txt-att/data', split='val', get_scores=True)
        write_bleu(scores=scores, path=test_path, epoch=0)
        
            
if __name__ == "__main__":
  tf.app.run()
