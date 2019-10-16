from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

PICTURE_SIZE = 128

def make_square_image(im):
    width, height = im.size
    max_len = max(width, height)
    new_im = Image.new('L', (max_len, max_len), 0)
    new_im.paste(im, (int((max_len - width) / 2), int((max_len - height) / 2)))
    return new_im

def load_image():
    size = (PICTURE_SIZE,PICTURE_SIZE)
    img_list = []
    img_label = []
    base_path = r'D:\Program\98semester\VRDL\HW1\dataset\dataset\train'
    count = -1
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                filename = os.path.join(root,file)
                category_name = os.path.basename(root)
                im = Image.open(filename)
                width, height = im.size
                if width != height :
                    im = make_square_image(im)
                im=im.resize(size,Image.BILINEAR)
                imarray = np.array(im)
                img_list.append(imarray)
                img_label.append(count)
        count += 1
    
    img_arr = np.asarray(img_list)
    img_label = to_categorical(img_label)
    print("Image Loading Complete")

    import random
    temp = list(zip(img_arr, img_label))
    random.shuffle(temp)
    img_arr, img_label = zip(*temp)
    img_arr=np.asarray(img_arr)
    img_label=np.asarray(img_label)
    print("Random Complete")

    from sklearn.model_selection import train_test_split
    train_data, test_data, train_label, test_label = train_test_split(img_arr, img_label, test_size=0.2, random_state=42)
    print("Split Complete")

    return (train_data, train_label), (test_data, test_label)

(x_train, y_train), (x_test, y_test) = load_image()

x_train = x_train.reshape(x_train.shape[0], PICTURE_SIZE, PICTURE_SIZE, 1)
x_test = x_test.reshape(x_test.shape[0], PICTURE_SIZE, PICTURE_SIZE, 1)
#x_train = x_train.reshape(x_train.shape[0], 256*256)
#x_test = x_test.reshape(x_test.shape[0], 256*256)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, LeakyReLU
from keras import optimizers
'''
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)
'''
input_shape = (PICTURE_SIZE,PICTURE_SIZE,1)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(13, activation='softmax')
])
'''
model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(256*256,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(13, activation='softmax'))
'''
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005),
              metrics=['accuracy'])

model.summary()

train_history=model.fit(x_train,
                        y_train,  
                        epochs=20,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, y_test))

model.save('my_model.h5')

exit()