#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


plt.figure(figsize=(20,20))
img_folder=r'Defect/test/def_front'
for i in range(5):
    file = random.choice(os.listdir(img_folder))
    image_path= os.path.join(img_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)


# In[3]:


IMG_WIDTH=200
IMG_HEIGHT=200
def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name


# In[4]:


img_folder1=r'Defect/train'
img_data, class_name =create_dataset(img_folder1)


# In[5]:


img_folder2=r'Defect/test'
img_data_test,class_name_test=create_dataset(img_folder2)


# In[6]:


target_dict={k: v for v, k in enumerate(np.unique(class_name))}
target_dict


# In[7]:


target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]


# In[8]:


x1=np.array(img_data,np.float32)
print(type(x1))
print(x1.shape)
x_train=x1[:,:,:,1]


# In[9]:


x2=np.array(img_data_test,np.float32)
print(type(x2))
print(x2.shape)
x_test=x2[:,:,:,1]


# In[171]:


array11= np.zeros((3758,1),dtype=int)
array12=np.ones((2875,1),dtype=int)
y_train=np.append(array11,array12)
print(y_train.shape)


# In[172]:


array21= np.zeros((453,1),dtype=int)
array22=np.ones((262,1),dtype=int)
y_test=np.append(array21,array22)
print(y_test.shape)


# In[43]:


plt.gray() # B/W Images
plt.figure(figsize = (10,9))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])


# In[44]:


for i in range(5):
    print(y_train[i])


# In[174]:


print(x_train.min())
print(x_train.max())
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')


# Reshaping input data:


X_train = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)


# In[176]:


print(X_train.shape)
print(X_test.shape)


# In[177]:


from sklearn.cluster import MiniBatchKMeans
total_clusters = len(np.unique(y_test))

# Initialize the K-means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
#Fitting model to the training set
kmeans.fit(X_train)


# In[178]:


kmeans.labels_


# In[182]:


def retrieve_info(cluster_labels,y_train):
    reference_labels = {}
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels
   
print(reference_labels)

# In[184]:


reference_labels = retrieve_info(kmeans.labels_,y_train)
print(reference_labels)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]
print(number_labels)


# In[185]:


print(number_labels[:6633].astype('int'))
print(y_train[:6633])


# In[186]:


from sklearn.metrics import accuracy_score
print(accuracy_score(number_labels,y_train))


# In[187]:


def calculate_metrics(model,output):
    print('Number of clusters is {}'.format(model.n_clusters))
    print('Inertia : {}'.format(model.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(output,model.labels_)))


# In[188]:


from sklearn import metrics
cluster_number = [10,16,36,64,144,256]
for i in cluster_number:
    total_clusters = len(np.unique(y_test))
    kmeans = MiniBatchKMeans(n_clusters = i)
    kmeans.fit(X_train)
    kmeans.labels_
    reference_labels = retrieve_info(kmeans.labels_,y_train)
    print(reference_labels)
    calculate_metrics(kmeans,y_train)
    reference_labels = retrieve_info(kmeans.labels_,y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
    print('Accuracy score : {}'.format(accuracy_score(number_labels,y_train)))
    print('\n')


# In[190]:


print(kmeans.labels_.shape)
len(np.unique(kmeans.labels_))
def retrieve_info1(cluster_labels,y_train):
    reference_labels = {}
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        if np.bincount(y_train[index==1]).size !=0:
            num = np.bincount(y_train[index==1]).argmax()
            reference_labels[i] = num
        else:
            reference_labels[i] = num
    return reference_labels
reference_labels = retrieve_info1(kmeans.labels_,y_test)


# In[195]:


kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X_test)
calculate_metrics(kmeans,y_test)
reference_labels = retrieve_info1(kmeans.labels_,y_test)
print(len(reference_labels))
print(len(kmeans.labels_))


# In[196]:


kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X_test)
calculate_metrics(kmeans,y_test)
reference_labels = retrieve_info(kmeans.labels_,y_test)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]
print('Accuracy score : {}'.format(accuracy_score(number_labels,y_test)))
print('\n')


# In[197]:


centroids = kmeans.cluster_centers_
centroids.shape
centroids = centroids.reshape(256,200,200)


# In[198]:


plt.figure(figsize = (10,9))
bottom = 0.35
for i in range(16):
    plt.subplots_adjust(bottom)
    plt.subplot(4,4,i+1)
    plt.title('Number:{}'.format(reference_labels[i]),fontsize = 17)
    plt.imshow(centroids[i])


# In[211]:


image=plt.imread('cast_def_0_7.jpeg')
image.shape
from skimage import color
from skimage import io
image = color.rgb2gray(io.imread('cast_def_0_7.jpeg'))
image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
image=np.array(image)
image = image.astype('float32')
image = image.reshape(1,200*200)
print(image.shape)


# In[212]:


x_train = x_train.astype('float32')
x_train = x_train.reshape(6633,200*200)


# In[213]:


kmeans = MiniBatchKMeans(n_clusters=256)
kmeans.fit(x_train)


# In[214]:


reference_labels = retrieve_info(kmeans.labels_,y_train)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]


# In[215]:


predicted_cluster = kmeans.predict(image)


# In[216]:


number_labels[predicted_cluster]


# In[ ]:




