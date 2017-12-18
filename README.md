# im2txt-att
Attention-based Image Captioning
## Prerequisite
Python 2.7
TensorFlow 1.0 or greater
Numpy
Scipy
matplotlib
hickle
Pillow
scikit-image
## Prepare the Training Data
### Download MSCOCO image captioning data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P data/
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P image/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P image/
### Download the VGGnet19 Chekpoint
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/
## Preprocess Data
$ python process_MSCOCO.py
## Train the Model
$ python train.py
## Evaluate the Model
$ python evaluation.py
