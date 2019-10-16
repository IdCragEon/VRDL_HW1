import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

PICTURE_SIZE = 128
flag = ['bedroom', 'coast', 'forest', 'highway', 'insidecity', 
        'kitchen', 'livingroom', 'mountain', 'office', 
        'opencountry', 'street', 'suburb', 'tallbuilding']

def make_square_image(im):
    width, height = im.size
    max_len = max(width, height)
    new_im = Image.new('L', (max_len, max_len), 0)
    new_im.paste(im, (int((max_len - width) / 2), int((max_len - height) / 2)))
    return new_im

def load_image():
    size = (PICTURE_SIZE,PICTURE_SIZE)
    img_list = []
    file_name = []
    base_path = r'D:\Program\98semester\VRDL\HW1\dataset\dataset\test'
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                file_name.append(file.split('.')[0])
                filename = os.path.join(root,file)
                category_name = os.path.basename(root)
                im = Image.open(filename)
                width, height = im.size
                if width != height :
                    im = make_square_image(im)
                im=im.resize(size,Image.BILINEAR)
                imarray = np.array(im)
                img_list.append(imarray)
    img_arr = np.asarray(img_list)
    print("Image Loading Complete")

    return img_arr, file_name


x_test, file_name = load_image()

print(x_test.shape)
x_test = x_test.reshape(x_test.shape[0], PICTURE_SIZE, PICTURE_SIZE, 1)
x_test = x_test.astype('float32')
x_test /= 255

print(x_test.shape)

from keras.models import load_model
from keras.models import Sequential

model = Sequential()
model = load_model('my_model.h5')
prediction = model.predict_classes(x_test)

import csv
with open('prediction.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for i in range(len(prediction)):
        print([file_name[i],flag[prediction[i]]])
        writer.writerow([file_name[i],flag[prediction[i]]])