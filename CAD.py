#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as bk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


image_width, image_height = 150, 150
train_set_dir = 'dataset/training_set/'
test_set_dir = 'dataset/test_set/'

b_size = 1
# Training Set Image Generator
#trainSetImageGenerator = ImageDataGenerator(rescale= 1. / 255,shear_range= 0.3, zoom_range=0.1, horizontal_flip=True)
trainSetImageGenerator = ImageDataGenerator(rescale= 1. /255)
# Test Set Image Generator
testSetImageGenerator = ImageDataGenerator(rescale= 1. /255)


# In[33]:


trainSetGenerator = trainSetImageGenerator.flow_from_directory(train_set_dir, 
                                                               target_size=(image_width, image_height), shuffle=False, batch_size=b_size, class_mode='binary')

testSetGenerator = testSetImageGenerator.flow_from_directory(test_set_dir, 
                                                             target_size=(image_width, image_height), batch_size=b_size, class_mode='binary')


# In[34]:


img = trainSetGenerator[0][0][0]
plt.imshow(img)
plt.show
print('Length training: {}'.format(len(trainSetGenerator)))
print('Length testing: {}'.format(len(testSetGenerator)))


# In[35]:


model = Sequential()


# In[36]:


model.add(Conv2D(32, kernel_size=(5,5), activation=keras.activations.relu))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), activation=keras.activations.relu))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation=keras.activations.relu))
model.add(Dropout(0.4))
#model.add(Dense(64, activation=keras.activations.relu))
#model.add(Dropout(0.5))
model.add(Dense(2, activation=keras.activations.softmax))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adam(), metrics=['accuracy'])


# In[37]:




no_of_ephoc = 4
model.fit_generator(trainSetGenerator,  epochs=no_of_ephoc, validation_data=testSetGenerator)


# In[38]:


validationSetImageGenerator = ImageDataGenerator(rescale= 1. /255)
valid_set_dir = 'data/test/'
validationSetGenerator = validationSetImageGenerator.flow_from_directory(valid_set_dir, 
                                                               target_size=(image_width, image_height),
                                                                shuffle=False, 
                                                                batch_size=b_size, class_mode='binary')


# In[46]:


img1 = validationSetGenerator[0][0][0]


# In[47]:


plt.imshow(img1)
plt.show()


# In[48]:


score = model.evaluate_generator(validationSetGenerator, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[54]:


image_to_predict = image.load_img('data/test/cats/103.jpg', target_size=(image_width, image_height))
image_to_predict = image.img_to_array(image_to_predict)
image_to_predict = np.expand_dims(image_to_predict, axis=0)


# In[57]:


pred = model.predict(image_to_predict)
pred_clas = model.predict_classes(image_to_predict)


# In[58]:


pred


# In[ ]:




