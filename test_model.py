#!/usr/bin/env python3
from keras.models import Sequential, load_model
import keras
import matplotlib.pyplot as plt
import numpy as np
import imageio
import scipy
import os

hex_key = {'30': '0', '31': '1', '32': '2', '33': '3', '34': '4', '35': '5', '36': '6',
'37': '7', '38': '8', '39': '9', '41': 'A', '42': 'B', '43': 'C', '44': 'D', '45': 'E',
'46': 'F', '47': 'G', '48': 'H', '49': 'I', '4A': 'J', '4B': 'K', '4C': 'L', '4D': 'M',
'4E': 'N', '4F': 'O', '50': 'P', '51': 'Q', '52': 'R', '53': 'S', '54': 'T', '55': 'U',
'56': 'V', '57': 'W', '58': 'X', '59': 'Y', '5A': 'Z', '61': 'a', '62': 'b', '63': 'c',
'64': 'd', '65': 'e', '66': 'f', '67': 'g', '68': 'h', '69': 'i', '6A': 'j', '6B': 'k',
'6C': 'l', '6D': 'm', '6E': 'n', '6F': 'o', '70': 'p', '71': 'q', '72': 'r', '73': 's',
'74': 't', '75': 'u', '76': 'v', '77': 'w', '78': 'x', '79': 'y', '7A': 'z'}

class_key = {}

classes = os.listdir('by_class')

data_pairs = []
for class_int, class_dir in enumerate(classes):
    class_key[class_int] = hex_key[class_dir.upper()]
    image_path = 'by_class/{}/train_{}/'.format(class_dir, class_dir)
    image_files = os.listdir(image_path)
    for d in [image_path + img for img in image_files]:
        data_pairs.append((d, class_int))

x_data = np.array(data_pairs)[::,0]
y_data = keras.utils.to_categorical(np.array(data_pairs)[::,1], len(classes))

model = load_model('models/model.hdf5')

for i in range(100):
    idx = np.random.randint(0, len(data_pairs))
    t1 = "Actual : " + str(class_key[data_pairs[idx][1]])
    im = imageio.imread(data_pairs[idx][0], pilmode='L')/255
    pred = np.argmax(model.predict(np.array([im.reshape(1,128,128)])))
    t2 = "Predict: " + str(class_key[pred])
    print(t1)
    print(t2)
    print("---------")
    plt.text(0, 10, t1)
    plt.text(0, 15, t2)
    plt.imshow(im)
    plt.pause(1)
    plt.cla()
