#!/usr/bin/env python
# coding: utf-8

# In[129]:


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


# In[130]:


model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)


# In[131]:


# model.summary()


# In[132]:


cat_file = open("imagenet1000_clsidx_to_labels.txt", "r")

contents = cat_file.read()
cat_dict = ast.literal_eval(contents)
cat_file.close()
print(len(cat_dict))


# In[133]:


def find_cat_in_dict(prediction_name):
    for cat_num, cat_name in cat_dict.items():
        if prediction_name in cat_name:
            return cat_num


# In[144]:


# Image titles
# image_titles = ['Goldfish', "Beer", 'Axolotl']

# Load images
img1 = load_img('images/oscar.jpg', target_size=(224, 224))
img2 = load_img('images/jeb.jpg', target_size=(224, 224))
img3 = load_img('images/ball.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data
preprocessed_images = preprocess_input(images)
image_titles = []
image_categories = []
for i in preprocessed_images:
    x = i.reshape((1, i.shape[0], i.shape[1], i.shape[2]))
    yhat = model.predict(x)
    label = decode_predictions(yhat)
    label = label[0][0]
    prediction_name = label[1].replace('_', ' ')
    img_title = (f'{prediction_name} ({int(math.ceil(label[2] * 100))}%)')
    image_titles.append(img_title)
    image_categories.append(find_cat_in_dict(prediction_name))


# In[145]:


def loss(output):
    return (output[0][image_categories[0]], output[1][image_categories[1]], output[2][image_categories[2]])


# In[146]:


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m


# In[151]:


from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

# Create Gradcam object
gradcam = Gradcam(model,
                  model_modifier=model_modifier,
                  clone=False)

# Generate heatmap with GradCAM
"""
cam = gradcam(loss,
              preprocessed_images,
              penultimate_layer=-1, # model.layers number
             )
"""
cam = gradcam(loss,
              preprocessed_images,
              penultimate_layer=-1,
             )
cam = normalize(cam)


# In[152]:


subplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.show()

f, bx = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    bx[i].imshow(images[i])
    bx[i].imshow(heatmap, cmap='jet', alpha=0.4) # overlay
plt.tight_layout()
plt.show()
