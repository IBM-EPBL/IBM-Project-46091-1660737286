#!/usr/bin/env python
# coding: utf-8

# Importing The ImageDataGenerator Library

# In[26]:


cd/content/drive/MyDrive/Project


# In[2]:


get_ipython().system('unzip /content/drive/MyDrive/Project/archive.zip')


# In[3]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-contrib-python')
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# In[5]:


train=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 rotation_range=180,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)


# In[6]:


train_dataset = train.flow_from_directory("/content/drive/MyDrive/Project/Dataset/Dataset/train_set",
                                          target_size=(128,128),
                                          batch_size = 32,
                                          class_mode = 'binary' )


# In[7]:


test_dataset = test.flow_from_directory("/content/drive/MyDrive/Project/Dataset/Dataset/test_set",
                                          target_size=(128,128),
                                          batch_size = 32,
                                          class_mode = 'binary' )


# In[8]:


#to define linear initialisation import sequential
from keras.models import Sequential
#to add layer import Dense
from keras.layers import Dense
#to create convolution kernel import convolution2D
from keras.layers import Convolution2D
#import Maxpooling layer
from keras.layers import MaxPooling2D
#import flatten layer
from keras.layers import Flatten
import warnings
warnings.filterwarnings('ignore')


# In[9]:


model = keras.Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())


# In[10]:


model.add(Dense(150,activation='relu'))

model.add(Dense(1,activation='sigmoid'))


# In[11]:


model.compile(loss = 'binary_crossentropy',
              optimizer = "adam",
              metrics = ["accuracy"])


# In[12]:


r = model.fit(train_dataset, epochs = 5, validation_data = test_dataset)


# In[13]:


predictions = model.predict(test_dataset)
predictions = np.round(predictions)


# In[14]:


predictions


# In[15]:


print(len(predictions))


# In[16]:


model.save("forest1.h5")


# In[17]:


#import load_model from keras.model
from keras.models import load_model
#import image class from keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#import numpy
import numpy as np
#import cv2
import cv2


# In[18]:


model=load_model("forest1.h5")


# In[19]:


def predictImage(filename):
  img1 = image.load_img(filename,target_size=(128,128))
  Y = image.img_to_array(img1)
  X = np.expand_dims(Y,axis=0)
  val = model.predict(X)
  print(val)
  if val == 1:
    print(" fire")
  elif val == 0:
      print("no fire")


# In[20]:


predictImage("/content/drive/MyDrive/Project/Dataset/Dataset/test_set/with fire/180802_CarrFire_010_large_700x467.jpg")



# In[21]:


get_ipython().system('pip install twilio')


# In[22]:


get_ipython().system('pip install playsound')


# In[27]:


#import opencv librariy
import cv2
#import numpy
import numpy as np
#import image function from keras
from keras.preprocessing import image
#import load_model from keras
from keras.models import load_model
#import client from twilio API
from twilio.rest import Client
#imort playsound package
from playsound import playsound


# In[28]:


get_ipython().system('pip install pygobject')


# In[29]:


account_sid='AC8807c3d8de8e072b574584a440eca826'
auth_token='4ce5c7134f5b2d8f43ddab4a8ed86109'
client=Client(account_sid,auth_token)
message=client.messages.create(
    body='Forest Fire is detected,stay alert',
    from_='+14254751023',
    to='+919345123472')
print(message.sid)

