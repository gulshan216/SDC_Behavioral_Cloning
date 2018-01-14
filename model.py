import csv
import cv2
import numpy as np
# lines = []
# with open('../driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)


images=[]
measurements=[]
# for line in lines:
#     sourcepath = line[0]
#     current_path = '../IMG/' + sourcepath.split('/')[-1]
#     image=cv2.imread(current_path)
#     measurement = float(line[3])
#     images.append(image)
#     measurements.append(measurement)

image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_13_465.jpg")
images.append(image)
measurements.append(0.0)
image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_15_509.jpg")
images.append(image)
measurements.append(-0.508827)
image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_31_876.jpg")
images.append(image)
measurements.append(0.359026)

images_new = np.array(images)
measurements_new=np.array(measurements)
print(images_new.shape)

from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5,input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,border_mode='valid',activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(16,5,5,border_mode='valid',activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

model.fit(images_new,measurements_new,shuffle=True,nb_epoch=8,validation_split=0.33)

model.save('model.h5')
exit()