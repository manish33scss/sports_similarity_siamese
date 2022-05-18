# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:23:06 2022

@author: ASUS
"""


# %% import 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense,Activation, Lambda, MaxPooling2D, Input,Add, Flatten, Dropout, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random
import fnmatch
import datetime
# %% gather dataset (byteTrack) and create anchor, positive, negative
LOAD_DATA = r"D:\Work\Codes\ByteTrack_yolov5\yolov5\f_play_d1\images"
count = 0
anchor_img = []
positive_img = []
neg_img = []
import cv2
pos_label = []
neg_label = []
count = 0
for im in sorted(os.listdir(LOAD_DATA), key=len):
    #print(im)
    PATH_DATA = os.path.join(LOAD_DATA, im)
    num_files = len(fnmatch.filter(os.listdir(PATH_DATA), '*.*'))
    size = int(num_files/2)
    file_name = []
    for j in os.listdir(os.path.join(LOAD_DATA, im)):
        count += 1
        img_p = os.path.join(PATH_DATA, j)
        #img = preprocess(img_p)
        
        file_name.append(img_p)
        counter = "_"+ str(count)
        print(count, img_p)
    
    count = 0
    #path = os.path.join(LOAD_DATA, im)
    #s.mkdir(path)
    anc_names = file_name[0: int(len(file_name)* .5)]
    pos_names = file_name[int(len(file_name)* .5):]
    #a_counter = 0
    #p_counter = 0
    for i in anc_names:
        print(i)
        #a_counter +=1
        img = cv2.imread(i)
        img = cv2.resize(img, (100,100))
        img = img / 255.0
        anchor_img.append(img)
        

        #img_name = Path(i).stem
        #shutil.copy(i, ANCH_PATH)
        #src= os.path.join(ANCH_PATH, img_name+".jpg")
        #if exists(src):
            #anc_img = cv2.imread(src)
            # a_name = "ANC_"+img_name+ str(a_counter)+".jpg"
            # #head, tail = os.path.split(i)
            # dst = os.path.join(ANCH_PATH, a_name)
            # anch_list = []
            # os.rename(src, dst)

        
    for j in pos_names:
      print(j)
        #p_counter += 1 
        #img_name = Path(j).stem
      jmg = cv2.imread(j)
      img = cv2.resize(jmg, (100,100))
      img = img / 255.0
      positive_img.append(img)
      pos_label.append(0)
      
      
neg_data = r"D:\Work\Codes\ByteTrack_yolov5\yolov5\f_play_d1\Negative"

for file in os.listdir(neg_data):
    #print(os.path.join(neg_data, file))
    img = cv2.imread(os.path.join(neg_data, file))
    img = cv2.resize(img, (100,100))
    img = img / 255.0

    neg_img.append(img)
    neg_label.append(1)
    
#print(neg_img[:5], anchor_img[:5], positive_img[:5])

# 




# %%visualize dataset
def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()
this_pair = 810




# show images at this index
# show_image(pair1[0][0][this_pair])
# show_image(pair1[0][1][this_pair])
# # print the label for this pair
# print(pair1[0][2][this_pair])

# images = []
# for i in range(5):
#     # grab the current image pair and labe
#   imageA = dataset2[i][0]
#   imageB = dataset2[i][1]
#   label = labels[i]
#     # to make it easier to visualize the pairs and their positive or
#     # negative annotations, we're going to "pad" the pair with four
#     # pixels along the top, bottom, and right borders, respectively
#   output = np.zeros((36, 60), dtype="uint8")
#   pair = np.hstack([imageA, imageB])
#   plt.imshow(pair)
#   #plt.imshow(imageB)
#   print(label)
#   break

# %% create pairs
# dataset2 = []

# # for pairs [ (anchor, positive, label) and anchor, negative, label ] alternately
# labels = []
# for idxA in range(len(anchor_img)):
#         # grab the current image and label belonging to the current
#         # iteration
#         currentImage = anchor_img[idxA]
#         label = pos_label[idxA]
#         # randomly pick an image that belongs to the *same* class
#         # label
#         #idxB = np.random.choice(idx[label])
#         posImage = positive_img[idxA]
#         # prepare a positive pair and update the images and labels
#         # lists, respectively
#         dataset2.append([currentImage, posImage])
#         labels.append([1])
 
#         #negIdx = np.where(labels != label)[0]
#         negImage = neg_img[idxA]
#         # prepare a negative pair of images and update our lists
#         dataset2.append([currentImage, negImage])
#         labels.append([0])


# for contrastive loss where only two pairs are used


image1 = [] # all anchor images
image2 = [] # both positive and negative images
labels_ = [] # all labels


for idxA in range(len(anchor_img)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = anchor_img[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        #idxB = np.random.choice(idx[label])
        posImage = positive_img[idxA]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        image1.append(currentImage)
        image2.append(posImage)
        labels_.append(0)
        #dataset2.append([currentImage, posImage])
        #labels.append([1])
 
        #negIdx = np.where(labels != label)[0]
        negImage = neg_img[idxA]
        # prepare a negative pair of images and update our lists
        image1.append(currentImage)
        image2.append(negImage)
        labels_.append(1)




# prepare train and test sets
image1 = np.asarray(image1, dtype= "float32")
image2 = np.asarray(image2, dtype= "float32")
labels_ = np.asarray(labels_, dtype = "float32")
#labels_ = labels_.astype(float32)


tr_im1 = image1[int(len(image1)/4):]
ts_im1 = image1[:int(len(image1)/4)]

tr_im2 = image2[int(len(image2)/4):]
ts_im2 = image2[:int(len(image2)/4)]

tr_label = labels_[int(len(image1)/4):]
ts_label = labels_[:int(len(image1)/4)]

#visuallize 



# fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
# ax = ax.flatten()
# index = 0
# for i in range(0,6):
#   ax[index].imshow(tr_im1[i])
#   index +=1
#   ax[index+5].imshow(tr_im2[i])
# plt.tight_layout(pad=0.4)
# plt.show()


# TRAIN_LABEL = TRAIN_LABEL.astype('float32')
# TEST_LABEL = TEST_LABEL.astype('float32')
# normalize values
#train_images = train_images / 255.0
#test_images = test_images / 255.0

# create pairs on train and test sets
# tr_pairs, tr_y = TRAIN_DATA, TRAIN_LABEL
# ts_pairs, ts_y = TEST_DATA, TEST_LABEL

#tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
#ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)


# visualize 
# show_image(tr_pairs[:,0][0])
# show_image(tr_pairs[:,0][1])

# show_image(tr_pairs[:,1][0])
# show_image(tr_pairs[:,1][1])

# %%create embedding model

# %% first model
def initialize_base_network():
    inp = Input(shape=(100,100,3))
    c1 = Conv2D(64, (10,10), activation = "relu")(inp)
    m1 = MaxPooling2D (64, (2,2), padding= "same")(c1)
    
    
    c2 = Conv2D(128, (7,7), activation = "relu")(m1)
    m2 = MaxPooling2D (64, (2,2), padding= "same")(c2)
    
    c3 = Conv2D(128, (4,4), activation = "relu")(m2)
    m3 = MaxPooling2D (64, (2,2), padding= "same")(c3)
    
    c4 = Conv2D(256, (4,4), activation = "relu")(m3)
    drp1 = Dropout(0.2)(c4)
    f1 = Flatten()(drp1)
    d1 = Dense(4096, activation = "sigmoid") (f1)
    return Model(inputs=[inp], outputs=[d1], name = "embeddings")


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# %% Resnet50 embedding model

def embeddings():
      input = Input(shape=(100,100,3),name = 'base_input')
      x = Conv2D(64, 7, activation='relu' ,padding ='same')(input)
      x = Conv2D(64,1, activation = 'relu' )(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      it1 = MaxPooling2D((3,3))(x)
    
      x = Conv2D(64, 5, padding ='same')(it1)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(64, 5, padding ='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Add()([x,it1])
      it2 = Activation('relu')(x)
    
      x = Conv2D(64, 3, padding ='same')(it2)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(64, 3, padding ='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Add()([x,it2])
      it2 = Activation('relu')(x)
    
      x = GlobalAveragePooling2D()(x)
    
      output = Dense(11, activation='relu')(x)
    
      return Model(inputs=input, outputs = output)
    #return Model(inputs=[base_model.input], outputs=[d1], name = "embeddings")

# %% Functions

# def euclidean_distance(vects):
#     x, y = vects
#     sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))


# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)

base_network = embeddings()


# %%create distance layer
# Build distance layer : template 

class L1dist(Layer):
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def call (self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
l1 = L1dist()

# %% siamese_model
def make_siamese():
    input_img = Input(name= "input_img", shape= (100,100,3))
    val_img = Input (name = "val_img", shape= (100,100,3))
    
    #siamese_layer = L1dist()
    #siamese_layer._name = 'distance'
    #distances = siamese_layer(
    vect_output_a =    base_network(input_img)
    vect_output_b=  base_network(val_img)

    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
    return   Model(inputs = [input_img, val_img], outputs = output, name='SiameseNetwork')

siamese_model = make_siamese()


# %%loss function ?
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer='adam')
siamese_model.summary()
# %%create training loop 
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

csv_file = 'training.csv'

history = siamese_model.fit([tr_im1, tr_im2], tr_label, 
                    epochs=120, batch_size=16, 
                    validation_data=([ts_im1, ts_im2], ts_label),
                    validation_split = 0.2,
                    verbose = 2,         
                    callbacks=[EarlyStopping(
                        patience=10,
                        min_delta=0.05,
                        baseline=0.8,
                        mode='min',
                        monitor='val_loss',
                        restore_best_weights=True,
                        verbose=1),
                        CSVLogger(csv_file),
#                        ReduceLROnPlateau(monitor='val_loss', 
#                                       factor=0.2, verbose=1,
#                                       patience=1, min_lr=0.001)
                    ])


# %% save model

siamese_model.save('meesi_siamese2.h5')

print(history.history.keys())


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()


np.unique(ts_label, return_counts=True)

# %% test_model

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for i in range(0,6):
  ax[index].imshow(ts_im1[i])
  index +=1
  ax[index+5].imshow(ts_im2[i])
plt.tight_layout(pad=0.4)
plt.show()


predictions = siamese_model.predict([ts_im1, ts_im2])
