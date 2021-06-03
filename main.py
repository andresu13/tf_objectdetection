import keras
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.utils.vis_utils import plot_model
#print('Keras version:', keras.__version__)
#print('TensorFlow version:', tf.__version__)
#import sys
#print(sys.version)

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
#conv_base.summary()
plot_model(conv_base, to_file='model_plot.png', show_shapes=True, show_layer_names=True)