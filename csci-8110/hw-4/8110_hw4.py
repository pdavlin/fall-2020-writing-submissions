#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D


import tensorflow as tf
import csv

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
import matplotlib.pyplot as plt
from datetime import datetime


# In[2]:


# necessary for my 3080 GPU to ensure it performs as expected 
# and doesn't weirdly run out of memory at the START of the run

# I'm not sure if this impacts Colab, may need to comment it out

# this setting is only applied once *per session*, subsequent runs won't do anything?
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[3]:


# define discriminator as in assignment description
def discriminator_impl(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(32, (5,5), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (5,5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (5,5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (5,5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.trainable = False
    return model

def generator_impl():
    model = Sequential()

    num_units = 7 * 7 * 192

    model.add(Dense(num_units, input_dim=100))
    
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7,7,192)))
    
    model.add(Dropout(0.4))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(96, (5,5), strides=(1), padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D(size=(2, 2)))

    #DeConv
    model.add(Conv2DTranspose(48, (5,5), strides=(1), padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(24, (5,5), strides=(1), padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, (5,5), strides=(1), padding='same'))
    
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model


# In[4]:


def load_images():
	# load mnist dataset
	(train_data, _), (_, _) = load_data()
	X = expand_dims(train_data, axis=-1)
	X = X.astype('float32')
	X = X / 255.0
	return X

# select real samples
def get_real_data(dataset, num_samples):
	ix = randint(0, dataset.shape[0], num_samples)
	X = dataset[ix]
	# label as real
	y = ones((num_samples, 1))
	return X, y

# use the generator to generate fresh images
def get_fake_data(generator, dim_value, num_samples):
	x_input = randn(dim_value * num_samples)
	x_input = x_input.reshape(num_samples, dim_value)
	# based on random input, generate images from
	X = generator.predict(x_input)
	# label as fake samples
	y = zeros((num_samples, 1))
	return X, y


# In[5]:


def plot_results(epoch, generator, discriminator, dataset, dim_val):
	x_fake, _ = get_fake_data(generator, dim_val, 16)
	for i in range(16):
		plt.subplot(4, 4, 1 + i)
		plt.axis('off')
		plt.imshow(x_fake[i, :, :, 0], cmap='gray')
	# save plot to file
	filename = '%d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

def write_to_csv(filename, loss_list):
	with open(filename, 'a+', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)    
		csvwriter.writerow(loss_list)


# In[11]:


# train the generator and discriminator
def train(generator, discriminator, gan, dataset, dim_value):
	batch_size = 256
	now = datetime.now()
	dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
	filename = f'loss-{dt_string}.csv'

	#for every epoch...
	for i in range(600):
		# ... do all the training in the batch
		print('  running epoch %d...' % (i+1))
		for j in range(256):
			# pick random images from MNIST dataset
			x_real, y_real = get_real_data(dataset, 128)
			# generate fake images
			x_fake, y_fake = get_fake_data(generator, dim_value, 128)
			# combine two sets of images into one set of size 256
			x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
			# get discriminator loss
			discriminator_loss, _ = discriminator.train_on_batch(x, y)
			# set up input for generator
			x_input = randn(dim_value * batch_size)
			x_input = x_input.reshape(batch_size, dim_value)
			# label samples as real, for testing
			y_input = ones((batch_size, 1))
			# get gan loss
			gan_loss = gan.train_on_batch(x_input, y_input)
			# write loss values to CSV file for analysis later
			write_to_csv(filename, [float('{:,.3f}'.format(discriminator_loss)), float('{:,.3f}'.format(gan_loss))])
		# evaluate the model performance, sometimes
		if (i + 1) in [1,10,20] or ((i + 1) >= 100 and (i + 1) % 50 == 0):
			plot_results(i, generator, discriminator, dataset, dim_value)
	return loss_array


# In[12]:


print('defining discriminator')
discriminator = discriminator_impl()
discriminator.trainable = False
print('defining generator')
generator = generator_impl()
print('defining gan')
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
opt = Adam(lr=0.0002, beta_1=0.5)
gan.compile(loss='binary_crossentropy', optimizer=opt)

print('loading images from MNIST set')
dataset = load_images()
print('starting training...')
train(generator, discriminator, gan, dataset, 100)
print('training complete')

