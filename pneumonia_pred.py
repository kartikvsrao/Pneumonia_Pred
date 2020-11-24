import numpy as np
import seaborn as sea
import random
import os
import glob
import random

from sklearn.utils import shuffle
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sea.set_style("white") 



training_path = "chest_xray/chest_xray/train"
validation_path = "chest_xray/chest_xray/val"
testing_path = "chest_xray/chest_xray/test"



train_pneumonia_paths = glob.glob(os.path.join(os.path.join(training_path, "PNEUMONIA"), "*.jpeg"))
train_normal_path = glob.glob(os.path.join(os.path.join(training_path, "NORMAL"), "*.jpeg"))    


validation_pneumonia_path = glob.glob(os.path.join(os.path.join(validation_path, "PNEUMONIA"), "*.jpeg"))
validation_normal_path = glob.glob(os.path.join(os.path.join(validation_path, "NORMAL"), "*.jpeg"))  

test_pneumonia_path = glob.glob(os.path.join(os.path.join(testing_path, "PNEUMONIA"), "*.jpeg"))
test_normal_path = glob.glob(os.path.join(os.path.join(testing_path, "NORMAL"), "*.jpeg"))  

    
figure = plt.figure(constrained_layout = True, figsize=(10, 9))
grids = gridspec.GridSpec(nrows=3, ncols=3, figure=figure)
for i in range(9):
    y_ax, x_ax = i//3, i%3 
    a = figure.add_subplot(grids[y_ax,x_ax])
    file_name = os.path.basename(train_pneumonia_paths[i*10])
    img = tf.io.read_file(train_pneumonia_paths[i*10])
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [256,256], method="bicubic")
    img = tf.cast(img, tf.float32)  
    a.imshow(tf.cast(tf.squeeze(img), tf.uint8),
              cmap="summer", aspect="auto", vmin=0, vmax=255)
    a.axis("auto")
    a.title.set_text(file_name)
plt.suptitle("Pneumonia", fontsize = 16, y=1.05)




figure = plt.figure(constrained_layout = True, figsize=(10, 9))
grids = gridspec.GridSpec(nrows=3, ncols=3, figure=figure)
for i in range(9):
    y_ax, x_ax = i//3, i%3 
    a = figure.add_subplot(grids[y_ax,x_ax])
    file_name = os.path.basename(train_normal_path[i*10])
    img = tf.io.read_file(train_normal_path[i*10])
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [256,256], method="bicubic")
    img = tf.cast(img, tf.float32)  
    a.imshow(tf.cast(tf.squeeze(img), tf.uint8),
              cmap="summer", aspect="auto", vmin=0, vmax=255)
    a.axis("off")
    a.title.set_text(file_name)
plt.suptitle("Normal", fontsize = 16, y=1.05)


print("Normal training count {}".format(len(train_normal_path)))
print("Pneumonia training count {}".format(len(train_pneumonia_paths)))
print("Normal to Pnemonia ratio {:.1f}:{:.1f}".format((len(train_pneumonia_paths)+len(train_normal_path))/len(train_pneumonia_paths),(len(train_pneumonia_paths)+len(train_normal_path))/len(train_normal_path)))



figure, axle = plt.subplots(constrained_layout = True,nrows=6, ncols=5, figsize=(10,14))
idx = [99, 128, 150, 175, 200, 225]

for i in range(6):     
    img = tf.io.read_file(train_pneumonia_paths[idx[i]])
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [256,256], method="bicubic")
    img = tf.cast(img, tf.float32)    
    
    smpl = []
    smpl.append(img)
    
    #Random rotate
    t = tf.random.uniform([], minval=-0.3,maxval=0.3,dtype=tf.float32)
    t_mat = tf.convert_to_tensor([[tf.math.cos(t),-tf.math.sin(t)],[tf.math.sin(t),tf.math.cos(t)]])


    coordinates = tf.stack([tf.repeat(tf.range(-256/2, 256/2), len(tf.range(-256/2, 256/2))), tf.tile(tf.range(-256/2, 256/2), [len(tf.range(-256/2, 256/2))])], axis=1)
    
    t_coordinates = tf.transpose(tf.linalg.matmul(t_mat,tf.transpose(coordinates)))
    
    off = tf.stack([tf.repeat(256/2, len(t_coordinates)), tf.repeat(256/2, len(t_coordinates))], axis=1)
    
    t_coordinates = tf.math.add(t_coordinates, off)
    t_coordinates = tf.clip_by_value(t_coordinates, 0, 256-1)
    
    t_pixels = tf.gather_nd(img, tf.cast(t_coordinates, tf.int32))    
    t_img = tf.reshape(t_pixels, [256,256,1])
    
    smpl.append(t_img)
    
    #Random shift
    
    height1 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)
    height2 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)    
    width1 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)    
    width2 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)
    t_img = tf.image.crop_to_bounding_box(img, height1, width1,
                                              256-height1-height2, 256-width1-width2)
    t_img = tf.image.resize(t_img, size=[256, 256],
                                            method="bicubic")  
    
    smpl.append(tf.reshape(t_img, [256, 256, 1]))
    
    #Random Zoom

    
    sf = tf.random.uniform([], minval=0.8,
                                        maxval=1.3,
                                        dtype=tf.float32)
    res_sz = tf.cast(sf*256, tf.int32)
    s_img = tf.image.resize(img, size=[res_sz,res_sz],
                                          method="bicubic")
    t_img = tf.image.resize_with_crop_or_pad(s_img, 256, 256)    
    smpl.append(tf.reshape(t_img, [256, 256, 1]))
    
    #Random Contrast

    
    contrst = tf.random.uniform([], minval=0.7,
                                        maxval=1.3,
                                        dtype=tf.float32)

    t_img = tf.image.adjust_contrast(img, contrst)  
    t_img = tf.minimum(t_img, 255)
    t_img = tf.maximum(t_img, 0)
    smpl.append(tf.reshape(t_img, [256, 256, 1]))
    
    
    for s in range(5):
        axle[i][s].imshow(tf.cast(tf.squeeze(smpl[s]), tf.uint8),
                  cmap="summer", aspect="auto", vmin=0, vmax=255)
        axle[i][s].axis("off")

axle[0][0].title.set_text("Original")
axle[0][1].title.set_text("Random Rotation")
axle[0][2].title.set_text("Random Shift")
axle[0][3].title.set_text("Random Zoom")
axle[0][4].title.set_text("Random Contrast")


tempp = np.array(train_pneumonia_paths)
tempn = np.array(train_normal_path)

temp_train_x = np.concatenate([tempp, tempn])


temp_train_y = np.zeros(len(temp_train_x), dtype=np.float32)
temp_train_y[:len(tempp)] = 1

for s in range(50):
    temp_train_x, temp_train_y = shuffle(temp_train_x, temp_train_y)
    
tempp = np.array(validation_pneumonia_path)
tempn = np.array(validation_normal_path)


temp_validation_x = np.concatenate([tempp, tempn])


temp_validation_y = np.zeros(len(temp_validation_x), dtype=np.float32)
temp_validation_y[:len(tempp)] = 1


for s in range(50):
    temp_validation_x, temp_validation_y = shuffle(temp_validation_x, temp_validation_y)
    

tempp = np.array(test_pneumonia_path)
tempn = np.array(test_normal_path)


temp_test_x = np.concatenate([tempp, tempn])


temp_test_y = np.zeros(len(temp_test_x), dtype=np.float32)
temp_test_y[:len(tempp)] = 1


for s in range(50):
    temp_test_x, temp_test_y = shuffle(temp_test_x, temp_test_y)
    
    
training = tf.data.Dataset.from_tensor_slices((temp_train_x, temp_train_y))
validation = tf.data.Dataset.from_tensor_slices((temp_validation_x, temp_validation_y))
testing = tf.data.Dataset.from_tensor_slices((temp_test_x, temp_test_y))

training_sz = tf.data.experimental.cardinality(training).numpy()
validation_sz = tf.data.experimental.cardinality(validation).numpy()
testing_sz = tf.data.experimental.cardinality(testing).numpy()
print("Training dataset count   : {}".format(training_sz))
print("Validation dataset count : {}".format(validation_sz))
print("Testing dataset count       : {}".format(testing_sz))

e = 120
bs = 64

def read_process(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [256,256], method="bicubic")
    img = tf.cast(img, tf.float32)    
    img = (img - 127.5) / 255.0
    return img, label

def augment(img, label):    
  
    
    
    rand_no = random.randint(1,4)
    
    if rand_no == 1:
        t = tf.random.uniform([], minval=-0.3,
                                  maxval=0.3,
                                  dtype=tf.float32)
        t_mat = tf.convert_to_tensor([[tf.math.cos(t),
                                          -tf.math.sin(t)],
                                          [tf.math.sin(t),
                                           tf.math.cos(t)]])
        coordinates = tf.stack([tf.repeat(tf.range(-256/2, 256/2), len(tf.range(-256/2, 256/2))), 
                           tf.tile(tf.range(-256/2, 256/2), [len(tf.range(-256/2, 256/2))])], axis=1)

        t_coordinates = tf.transpose(tf.linalg.matmul(t_mat,
                                                tf.transpose(coordinates)))

        off = tf.stack([tf.repeat(256/2, len(t_coordinates)), 
                           tf.repeat(256/2, len(t_coordinates))], axis=1)

        t_coordinates = tf.math.add(t_coordinates, off)
        t_coordinates = tf.clip_by_value(t_coordinates, 0, 256-1)

        t_pixels = tf.gather_nd(img, tf.cast(t_coordinates, tf.int32))    
        t_img = tf.reshape(t_pixels, [256,256,1])
        
    elif rand_no == 2:
        
        height1 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)
        height2 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)    
        width1 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)    
        width2 = tf.random.uniform([], minval=0, maxval=40, dtype=tf.int32)
        t_img = tf.image.crop_to_bounding_box(img, height1, width1,
                                                  256-height1-height2, 256-width1-width2)
        t_img = tf.image.resize(t_img, size=[256, 256],
                                                method="bicubic")  
        t_img =  tf.reshape(t_img, [256, 256, 1])
    
    elif rand_no ==3:
        sf = tf.random.uniform([], minval=0.8,
                                        maxval=1.3,
                                        dtype=tf.float32)
        res_sz = tf.cast(sf*256, tf.int32)
        s_img = tf.image.resize(img, size=[res_sz,res_sz],
                                              method="bicubic")
        t_img = tf.image.resize_with_crop_or_pad(s_img, 256, 256)    
        t_img =  tf.reshape(t_img, [256, 256, 1])
    
    else:
        contrst = tf.random.uniform([], minval=0.7,
                                        maxval=1.3,
                                        dtype=tf.float32)

        t_img = tf.image.adjust_contrast(img, contrst)  
        t_img = tf.minimum(t_img, 1)
        t_img = tf.maximum(t_img, -1)
        t_img = tf.reshape(t_img, [256, 256, 1])
        
        
    
    
    

    return t_img, label  

training = training.map(read_process,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
training = training.map(augment,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
training = training.repeat().shuffle(1024).batch(bs) \
                .prefetch(tf.data.experimental.AUTOTUNE)

validation = validation.map(read_process,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
validation = validation.batch(bs) \
                .prefetch(tf.data.experimental.AUTOTUNE)

testing = testing.map(read_process,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
testing = testing.batch(len(temp_test_x))



input_t = Input(shape=(256,256,1))

x1 = Conv2D(32,(3,3), padding="same")(input_t)
x1 = BatchNormalization()(x1)
x1 = Activation("relu")(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = SpatialDropout2D(0.3)(x1)

x2 = Conv2D(32,(3,3), padding="same")(x1)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = MaxPooling2D(pool_size=(2,2))(x2)
x2 = SpatialDropout2D(0.3)(x2)

x3 = Conv2D(64,(3,3), padding="same")(x2)
x3 = BatchNormalization()(x3)
x3 = Activation("relu")(x3)
x3 = MaxPooling2D(pool_size=(2,2))(x3)
x3 = SpatialDropout2D(0.3)(x3)

x4 = Conv2D(64,(3,3), padding="same")(x3)
x4 = BatchNormalization()(x4)
x4 = Activation("relu")(x4)
x4 = MaxPooling2D(pool_size=(2,2))(x4)
x4 = SpatialDropout2D(0.3)(x4)

x5 = Conv2D(128,(3,3), padding="same")(x4)
x5 = BatchNormalization()(x5)
x5 = Activation("relu")(x5)
x5 = MaxPooling2D(pool_size=(2,2))(x5)
x5 = SpatialDropout2D(0.3)(x5)

x6 = Conv2D(256,(3,3), padding="same")(x5)
x6 = BatchNormalization()(x6)
x6 = Activation("relu")(x6)
x6 = SpatialDropout2D(0.3)(x6)

x7 = Conv2D(128,(1,1), padding="same")(x6)
x7 = BatchNormalization()(x7)
x7 = Activation("relu")(x7)
x7 = SpatialDropout2D(0.3)(x7)

x8 = Conv2D(256,(3,3), padding="same")(x7)
x8 = BatchNormalization()(x8)
x8 = Activation("relu")(x8)
x8 = SpatialDropout2D(0.3)(x8)

x9 = Conv2D(128,(1,1), padding="same")(x8)
x9 = BatchNormalization()(x9)
x9 = Activation("relu")(x9)
x9 = SpatialDropout2D(0.3)(x9)

x10 = Conv2D(256,(3,3), padding="same")(x9)
x10 = BatchNormalization()(x10)
x10 = Activation("relu")(x10)
x10 = SpatialDropout2D(0.3)(x10)

x11 = Conv2D(512,(3,3), padding="same")(x10)
x11 = BatchNormalization()(x11)
x11 = Activation("relu")(x11)
x11 = SpatialDropout2D(0.3)(x11)

x12 = Conv2D(256,(1,1), padding="same")(x11)
x12 = BatchNormalization()(x12)
x12 = Activation("relu")(x12)
x12 = SpatialDropout2D(0.3)(x12)

x13 = Conv2D(512,(3,3), padding="same")(x12)
x13 = BatchNormalization()(x13)
x13 = Activation("relu")(x13)
x13 = SpatialDropout2D(0.3)(x13)

x_pool = GlobalAveragePooling2D()(x13)

x_d1 = Dense(128)(x_pool)
x_d1 = BatchNormalization()(x_d1)
x_d1 = Activation("relu")(x_d1)
x_d1 = Dropout(0.3)(x_d1)

x_d2 = Dense(128)(x_d1)
x_d2 = BatchNormalization()(x_d2)
x_d2 = Activation("relu")(x_d2)
x_final = Dropout(0.3)(x_d2)

diagnose = Dense(1, activation="sigmoid")(x_final)

model = Model(inputs=input_t, outputs=diagnose)

model.summary()

optim = tf.keras.optimizers.Adam()
loss_bin = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)     
model.compile(loss=loss_bin,
            optimizer=optim,
            metrics=[tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")])

def decay(input):    
    learn_rate = 0.0003
    learn_rate = learn_rate * np.exp(-0.015*input)
    return learn_rate

learn_rate_sch = LearningRateScheduler(decay)
plt.figure(figsize=(8,5))
x = np.linspace(1,e + 1)
y = [decay(i) for i in x]
plt.plot(x,y);
plt.ylabel("Learning Rate")
plt.xlabel("Epoch")
plt.xticks(range(0,e+1,10));


spe = training_sz // bs
w = {0:3.0, 1:1.0}
history = model.fit(training, validation_data = validation,
                       steps_per_epoch = spe,
                       epochs = e,
                       class_weight = w,
                       verbose = 1, callbacks=[learn_rate_sch])

ep = np.linspace(1, e + 1, e)
figure, axle = plt.subplots(1, 1, figsize=(8,5))
sea.lineplot(x = ep, y = history.history["loss"], ax=axle, label="train");
sea.lineplot(x = ep, y = history.history["val_loss"], ax=axle, label="val");
axle.set_ylabel("Loss")
axle.set_xlabel("Epoch")
plt.xticks(range(0,e+1,10));

y_prob = model.predict(testing);
y_pred = [1 if x >= 0.5 else 0 for x in y_prob]
print(classification_report(temp_test_y, y_pred, digits = 2,target_names=["Normal", "Pneumonia"]))
