# Refer to code retrieved from https://github.com/yunjey/show-attend-and-tell
# Modification of the source code
# ==============================================================================

from PIL import Image
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

def _extract_feature(vgg_model_path, batch_size=32):
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for split in ['train', 'val', 'test']:
            anno_path = '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.annotations.pkl' % (split, split)
            save_path = '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)
            
            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
            
            for start, end in zip(range(0, n_examples, batch_size), 
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print('Processed %d %s features..' % (end, split))
            
            #use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print('Saved %s..' % (save_path))

def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1
            
    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx

def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length+2)).astype(np.int32)
    
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ')
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        #pad shord caption with the special null token '<NULL>' to make it a fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])
        captions[i, :] = np.asarray(cap_vec)
    print('Finished building caption vectors')
    return captions

def _build_vocab(annotations, threshold=1):
    print("Creating vocabulary.")
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ')
        for w in words:
            counter[w] +=1
        
        if len(caption.split(' ')) > max_len:
            max_len = len(caption.split(' '))
    
    vocab = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))
    
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3;
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    
    print('Max length of caption: ', max_len)
    return word_to_idx

def _build_caption_dataset(word_count_threshold=1, max_length=15):
    for split in ['train', 'val', 'test']:
        annotations = load_pickle('/home/yifan/PythonProjects/im2txt-att/data/%s/%s.annotations.pkl' % (split, split))
        if split == 'train':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, '/home/yifan/PythonProjects/im2txt-att/data/%s/word_to_idx.pkl' % split)
            
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.captions.pkl' % (split, split))
        
        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.file.names.pkl' % (split, split))
        
        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.image.idxs.pkl' % (split, split))
        
        #prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, '/home/yifan/PythonProjects/im2txt-att/data/%s/%s.references.pkl' % (split, split))
        print('Finished building %s caption dataset' % split)

def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        if tf.gfile.Exists(annotation['file_name']):
            data += [annotation]
    
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print("The number of captions before deletion: %d" %len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print("The number of captions after deletion: %d" %len(caption_data))
    return caption_data

def _resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def _resize_image_files():
    splits = ['train', 'val']
    for split in splits:
        folder = '/home/yifan/PythonProjects/im2txt-att/image/%s2014' %split
        resized_folder = '/home/yifan/PythonProjects/im2txt-att/image/%s2014_resized/' %split
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print('Start resizing %s images.' %split)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = _resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print('Resized images: %d/%d' %(i, num_images))


def main():
    #batch size for extracting feature vectors from vggnet
    batch_size = 32
    #maximum length of caption. If caption is longer than max length, deleted
    max_length = 15
    #if word occurs less than threshold in training set, the word index is unknown token
    word_count_threshold = 1
    #vgg model load path
    vgg_model_path = '/home/yifan/PythonProjects/im2txt-att/data/imagenet-vgg-verydeep-19.mat'
    
    
    #resize MSCOCO images into 224 * 224 size to fit Vgg
    _resize_image_files()
    
    #process training set and validation set
    train_dataset = _process_caption_data(caption_file='/home/yifan/PythonProjects/im2txt-att/data/annotations/captions_train2014.json',
                                          image_dir='/home/yifan/PythonProjects/im2txt-att/image/train2014_resized/',
                                          max_length=max_length)
    
    val_dataset = _process_caption_data(caption_file='/home/yifan/PythonProjects/im2txt-att/data/annotations/captions_val2014.json',
                                          image_dir='/home/yifan/PythonProjects/im2txt-att/image/val2014_resized/',
                                          max_length=max_length)
    
    # Redistribute the MSCOCO data as follows:
    #   val_dataset = 10% of mscoco_val_dataset (for validation during training).
    #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
    
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.2 * len(val_dataset))
    
    save_pickle(train_dataset, '/home/yifan/PythonProjects/im2txt-att/data/train/train.annotations.pkl')
    save_pickle(val_dataset[:val_cutoff], '/home/yifan/PythonProjects/im2txt-att/data/val/val.annotations.pkl')
    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), '/home/yifan/PythonProjects/im2txt-att/data/test/test.annotations.pkl')
    print('Finished processing MSCOCO caption data')
    
    #build caption datasets
    _build_caption_dataset(word_count_threshold=word_count_threshold, max_length=max_length)
    
    #extract conv5_3 feature vectors
    _extract_feature(vgg_model_path, batch_size=batch_size)
            
    
            
if __name__ == "__main__":
    main()