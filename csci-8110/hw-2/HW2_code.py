#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import ast
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# In[2]:


model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)


# In[3]:


cat_file = open("imagenet1000_clsidx_to_labels.txt", "r")

contents = cat_file.read()
cat_dict = ast.literal_eval(contents)
cat_file.close()
print(len(cat_dict))


# In[4]:


def find_cat_in_dict(prediction_name):
    for cat_num, cat_name in cat_dict.items():
        if prediction_name in cat_name:
            return cat_num


# In[5]:


# the tf-keras-vis library only takes an array,
# so we need to load our one image in as an array
input_title = 'axolotl'
img1 = load_img(f'images/{input_title}.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1)])

# get the top five predictions from our model
preprocessed_images = preprocess_input(images)
image_titles = []
image_categories = []
for i in preprocessed_images:
    x = i.reshape((1, i.shape[0], i.shape[1], i.shape[2]))
    y = model.predict(x)
    predictions = decode_predictions(y)
    predictions = predictions[0]
    for i in range(len(predictions)):
        prediction = predictions[i]
        prediction_name = prediction[1].replace('_', ' ')
        img_title = (f'{prediction_name} ({int(math.ceil(prediction[2] * 100))}%)')
        image_titles.append(img_title)
        image_categories.append(find_cat_in_dict(prediction_name))
print(image_categories)


# In[6]:


def loss(output):
    return (output[0][image_categories[counter]])


# In[7]:


def modifier(m):
    m.layers[-1].activation = tf.keras.activations.relu
    return m


# In[8]:


from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

# Create Gradcam object
gradcam = Gradcam(model,
                  model_modifier=modifier,
                  clone=False)


# In[9]:


subplot_args = { 'nrows': 1, 'ncols': 6, 'figsize': (20, 5),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
counter = 0
ax[0].set_title('Original image', fontsize=16)
ax[0].imshow(images[0])
for i, title in enumerate(image_titles):
    cam = gradcam(loss,
              preprocessed_images,
              penultimate_layer=-1,
             )
    cam = normalize(cam)
    ax[i+1].set_title(title, fontsize=16)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    ax[i+1].imshow(images[0])
    ax[i+1].imshow(heatmap, alpha=0.5)
    counter = counter + 1
plt.tight_layout()
plt.savefig(f'storage/{input_title}_output.png')
