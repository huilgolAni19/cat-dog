#!/usr/bin/env python
# coding: utf-8

# In[419]:


from __future__ import print_function
import cv2
import os
from os import walk
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[436]:


class ImageReader():
    base_ds_path = None
    f = None
    files_list = []
    def __init__(self, base_ds_path):
        self.base_ds_path= base_ds_path
        self.f = []

    def listImageFromBasePath(self):
        for (dirpath, dirnames, filenames) in walk(self.base_ds_path):
            for dir_name in dirnames:
                self.f.append(dir_name)

    def getImageList(self):
        if len(self.files_list) > 0:
            self.files_list.clear()
        for d in self.f:
            dir_path = '{}/{}{}'.format(os.path.abspath("."), self.base_ds_path, d)
            for (dirpath, dirnames, filenames) in walk(dir_path):
                for file_name in filenames:

                    image_path = '{}/{}'.format(dir_path,file_name)
                    self.files_list.append(image_path)
        return self.files_list

    def getBaseName(self, path):
        return os.path.basename(os.path.dirname(path))

    def getCategory(self, catName):
        if 'dog' in catName:
            return "Dog"
        else:
            return "Cat"


# In[437]:


train_set_path='data/train/'
test_set_path='data/test/'


# In[438]:


image_rdr_train = ImageReader(train_set_path)


# In[439]:


image_rdr_train.listImageFromBasePath()
img_train_path_list = image_rdr_train.getImageList()


# In[440]:


import numpy as np
np.set_printoptions(threshold=np.inf)

image_data_train = []
lable_data_train = []
for i, train_img_path in enumerate(img_train_path_list):
    img_data = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (150, 150))
    category_name = image_rdr_train.getCategory(image_rdr_train.getBaseName(train_img_path))
    image_data_train.append(img_data)
    lable_data_train.append(category_name)
#     print("Category: {}, shape: {}".format(category_name, img_data.shape))
#     plt.imshow(img_data)
#     plt.show()


# In[441]:


label_encoded_data_train = []
for label in lable_data_train:
    if 'Dog' in label:
        label_encoded_data_train.append(0)
    else:
        label_encoded_data_train.append(1)


# In[442]:


image_rdr_test = ImageReader(test_set_path)
image_rdr_test.listImageFromBasePath()
img_test_path_list = image_rdr_test.getImageList()

image_data_test = []
lable_data_test = []

for i, test_img_path in enumerate(img_test_path_list):
    img_data_t = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    img_data_t = cv2.resize(img_data_t, (150, 150))
    category_name_t = image_rdr_test.getCategory(image_rdr_test.getBaseName(test_img_path))
    #print(category_name_t)
    image_data_test.append(img_data_t)
    lable_data_test.append(category_name_t)



# In[443]:


label_encoded_data_test = []
for label in lable_data_test:
    if 'Dog' in label:
        label_encoded_data_test.append(0)
    else:
        label_encoded_data_test.append(1)


# In[444]:


image_data_train = np.asarray(image_data_train)
label_encoded_data_train = np.asarray(label_encoded_data_train)


image_data_test = np.asarray(image_data_test)
label_encoded_data_test = np.asarray(label_encoded_data_test)


# In[445]:


data_classes = ['Dog', 'Cat']


# In[446]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as bk
from tensorflow import keras


# In[447]:


no_of_epoch = 10
no_of_classes = 2
b_size = 5
img_row, img_col = 150 , 150

image_data_train = image_data_train.reshape(image_data_train.shape[0], img_row, img_col, 1)
image_data_test = image_data_test.reshape(image_data_test.shape[0], img_row, img_col, 1)
input_shape = (img_row, img_col, 1)

label_encoded_data_train = keras.utils.to_categorical(label_encoded_data_train, no_of_classes)
label_encoded_data_test = keras.utils.to_categorical(label_encoded_data_test, no_of_classes)


# In[448]:



model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation=keras.activations.relu))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), activation=keras.activations.relu))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation=keras.activations.relu))
model.add(Dropout(0.4))
model.add(Dense(no_of_classes, activation=keras.activations.relu))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adam(), metrics=['accuracy'])
#image_data_train[0]


# In[451]:


model.fit(image_data_train, label_encoded_data_train, batch_size=b_size, epochs=no_of_epoch, verbose=1, validation_data=(image_data_test, label_encoded_data_test))


# In[ ]:

'''
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-451-6a77121b9e14> in <module>
----> 1 model.fit(image_data_train, label_encoded_data_train, batch_size=b_size, epochs=no_of_epoch, verbose=1, validation_data=(image_data_test, label_encoded_data_test))

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
    726         max_queue_size=max_queue_size,
    727         workers=workers,
--> 728         use_multiprocessing=use_multiprocessing)
    729
    730   def evaluate(self,

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, **kwargs)
    222           validation_data=validation_data,
    223           validation_steps=validation_steps,
--> 224           distribution_strategy=strategy)
    225
    226       total_samples = _get_total_number_of_samples(training_data_adapter)

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2.py in _process_training_inputs(model, x, y, batch_size, epochs, sample_weights, class_weights, steps_per_epoch, validation_split, validation_data, validation_steps, shuffle, distribution_strategy, max_queue_size, workers, use_multiprocessing)
    545         max_queue_size=max_queue_size,
    546         workers=workers,
--> 547         use_multiprocessing=use_multiprocessing)
    548     val_adapter = None
    549     if validation_data:

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_v2.py in _process_inputs(model, x, y, batch_size, epochs, sample_weights, class_weights, shuffle, steps, distribution_strategy, max_queue_size, workers, use_multiprocessing)
    592         batch_size=batch_size,
    593         check_steps=False,
--> 594         steps=steps)
    595   adapter = adapter_cls(
    596       x,

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training.py in _standardize_user_data(self, x, y, sample_weight, class_weight, batch_size, check_steps, steps_name, steps, validation_split, shuffle, extract_tensors_from_dataset)
   2517           shapes=None,
   2518           check_batch_axis=False,  # Don't enforce the batch size.
-> 2519           exception_prefix='target')
   2520
   2521       # Generate sample-wise weight values given the `sample_weight` and

~/Documents/Simpli-Learn/Deep-Learning-With-Tensorflow/Projects/deeplearningenv/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/training_utils.py in standardize_input_data(data, names, shapes, check_batch_axis, exception_prefix)
    487       raise ValueError(
    488           'Error when checking model ' + exception_prefix + ': '
--> 489           'expected no data, but got:', data)
    490     return []
    491   if data is None:

ValueError: ('Error when checking model target: expected no data, but got:', array([[1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [1., 0.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 1.]], dtype=float32))
'''
