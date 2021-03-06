import time
import ast
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from datetime import datetime
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

from google.colab import drive
drive.mount('/content/drive')

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

cat_file = open("/content/drive/My Drive/8110/hw3/imagenet1000_clsidx_to_labels.txt", "r")

contents = cat_file.read()
cat_dict = ast.literal_eval(contents)
cat_file.close()
print(len(cat_dict))

def find_cat_in_dict(prediction_name):
  for cat_num, cat_name in cat_dict.items():
    if prediction_name in cat_name:
      return cat_num

def SENet_impl(init, ratio=16):
  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  filters = init.shape[channel_axis]
  se_shape = (1, 1, filters)

  x = GlobalAveragePooling2D()(init)
  x = Reshape(se_shape)(x)
  x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(x)
  x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(x)

  if K.image_data_format() == 'channels_first':
    x = Permute((3, 1, 2))(x)

  out = multiply([init, x])
  return out

# i = 0
# prevent original VGG16 layers from being trainable with new data
for layer in model.layers:
  layer.trainable = False
  # print(i, layer.name)
  # i = i + 1

from tensorflow.keras import Model

# save original model for comparison later
model_copy = model

for il in range(len(model_copy.layers) - 1):
  if il == 0:
    xl = model_copy.layers[il].output
  else:
    xl = model_copy.layers[il](xl)
  # locations of pooling: 3,6,10,14,18
  # can change location accordingly
  if il == 10:
    xl = SENet_impl(xl)

# reduced softmax layer (to reduce number of considered categories to 10)
xl = Dense(10,activation='softmax')(xl)

# define new model with SENet block
new_model= Model(model_copy.input,xl)

# compile new model
new_model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# new_model.summary()

def normalize_img(image, label):
  normalized = tf.cast(tf.image.resize(image, (224,224)), tf.float32) / 255., label
  return normalized

# load imagenette images for processing
builder = tfds.builder('imagenette/160px')
builder.download_and_prepare()
datasets = builder.as_dataset(as_supervised=True)
train_data,test_data = datasets['train'],datasets['validation']

# define and prefetch training data
training_dataset = train_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.batch(128)
training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# define and prefetch training data
validation_dataset = test_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(128)
validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# fit new data to train SENet and softmax layers
new_model.fit(training_dataset,
              validation_data=validation_dataset,
              batch_size=128,
              epochs=10)

# compare model summaries
# commented out to prevent clutter

# model.summary()
# new_model.summary()

# define reduced number of labels for classification and display
imagenette_labels = {
    0:  'tench',
    1:  'English_springer',
    2:  'casette_player',
    3:  'chain_saw',
    4:  'church',
    5:  'French_horn',
    6:  'garbage_truck',
    7:  'gas_pump',
    8:  'golf_ball',
    9:  'parachute'
}

# process individual images

from tensorflow.keras.preprocessing.image import load_img

# the tf-keras-vis library only takes an array,
# so we need to load our one image in as an array
input_title = 'horn'
img1 = load_img(f'/content/drive/My Drive/8110/hw3/inputs/{input_title}.JPEG', target_size=(224, 224))
images = np.asarray([np.array(img1)])

counter = 0
preprocessed_images = preprocess_input(images)
image_titles = []
image_categories = []
for i in preprocessed_images:
  x = i.reshape((1, i.shape[0], i.shape[1], i.shape[2]))
  y = new_model.predict(x)
  y = y[0]
  max_y = np.max(y)
  # define location where prediction is (value of 1 in array, all other values 0)
  max_loc = np.where(y == max_y)
  # get appropriate prediction label
  # some very lazy fitting of these predictions to my grad-CAM params from HW 2
  predictions = imagenette_labels.get(max_loc[0][0])
  prediction_name = predictions
  image_titles.append(prediction_name)
  image_categories.append(find_cat_in_dict(prediction_name))
print(prediction_name)

# get locations of layer where SENet output is (so I don't have to change these params manually)
layer_loc = 0
for layer in new_model.layers:
    if "multiply" in layer.name:
      break
    layer_loc = layer_loc + 1
print(f'SENet implementation ends at layer {layer_loc}. Previous block was {new_model.layers[layer_loc - 5].name} at loc  {layer_loc - 5}')

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

# convert the image to an array
img = img_to_array(img1)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# get the last layer of SENet_impl (multiply)
layer_loc = 0
for layer in new_model.layers:
    if "multiply" in layer.name:
      break
    layer_loc = layer_loc + 1

# get outputs from before the SENet block
featmap_model = Model(inputs=new_model.inputs, outputs=new_model.layers[layer_loc - 5].output)
feature_maps = featmap_model.predict(img)

print(f'Getting feature maps from {new_model.layers[layer_loc - 5].name} at layer {layer_loc - 5}')
pyplot.figure(figsize=(16,16))
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.savefig(f'/content/drive/My Drive/8110/hw3/outputs/{input_title}-pre-SENet-{new_model.layers[layer_loc - 5].name}-{datetime.now()}_output.png')

print(f'Getting feature maps from SENet output at layer {layer_loc}')

# get outputs from AFTER the SENet block
featmap_model = Model(inputs=new_model.inputs, outputs=new_model.layers[layer_loc].output)
feature_maps = featmap_model.predict(img)

pyplot.figure(figsize=(16,16))
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.savefig(f'/content/drive/My Drive/8110/hw3/outputs/{input_title}-post-SENet-{new_model.layers[layer_loc - 5].name}-{datetime.now()}_output.png')
