#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models


# In[5]:


# Define paths to your training dataset
train_data_dir = 'training'


# In[8]:


# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',  # 'binary' if you have two classes
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)



# In[9]:


# Load the InceptionV3 model without the final classification layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


# In[10]:


# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False


# In[11]:


# Create a new model on top of the pre-trained model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))


# In[12]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)


# In[15]:


# Save the trained weights
model.save('goal_inceptionv3.h5')


# In[ ]:




