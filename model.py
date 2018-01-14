import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
lines = []
with open('/home/carnd/SDC_Behavioral_Cloning/driving_log_track1_center_driving.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split

train_samples,validation_samples=train_test_split(lines,test_size=0.2)


# images=[]
# measurements=[]
# for line in lines:
#     sourcepath = line[0]
#     current_path = '../IMG/' + sourcepath.split('/')[-1]
#     image=cv2.imread(current_path)
#     measurement = float(line[3])
#     images.append(image)
#     measurements.append(measurement)

# image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_13_465.jpg")
# images.append(image)
# measurements.append(0.0)
# image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_15_509.jpg")
# images.append(image)
# measurements.append(-0.508827)
# image = cv2.imread("/home/gulshan216/Documents/CarND-Behavioral-Cloning-P3/IMG/center_2018_01_12_23_58_31_876.jpg")
# images.append(image)
# measurements.append(0.359026)


def generator(samples,batch_size=32):
	num_examples=len(samples)
	steer_correction=0.18
	while 1:
		shuffle(samples)
		for offset in range(0,num_examples,batch_size):
			X_batch=samples[offset:offset+batch_size]
			car_images=[]
			steer_ang=[]
			for row in X_batch:
				path1=row[0]
				path2=row[1]
				path3=row[2]
				center_img_path='/home/carnd/SDC_Behavioral_Cloning/IMG_track1_center_driving/'+path1.split('/')[-1]
				left_img_path='/home/carnd/SDC_Behavioral_Cloning/IMG_track1_center_driving/'+path2.split('/')[-1]
				right_img_path='/home/carnd/SDC_Behavioral_Cloning/IMG_track1_center_driving/'+path3.split('/')[-1]
				center_img=cv2.imread(center_img_path)
				center_img_flip=np.fliplr(center_img)
				left_img=cv2.imread(left_img_path)
				right_img=cv2.imread(right_img_path)
				center_ang = float(row[3])
				car_images.append(center_img)
				steer_ang.append(center_ang)
				car_images.append(left_img)
				steer_ang.append(center_ang+steer_correction)
				car_images.append(right_img)
				steer_ang.append(center_ang-steer_correction)
				car_images.append(center_img_flip)
				steer_ang.append(-center_ang)
				#car_images.extend(center_img,center_img_flip,left_img,right_img)
				#steer_ang.extend(center_ang,-center_ang,center_ang-steer_correction,center_ang+steer_correction)

			X_train = np.array(car_images)
			y_train = np.array(steer_ang)
			yield shuffle(X_train,y_train)

train_generator=generator(train_samples,batch_size=32)
validation_generator=generator(validation_samples,batch_size=32)

# images_new = np.array(images)
# measurements_new=np.array(measurements)
# print(images_new.shape)

from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model=Sequential()

model.add(Lambda(lambda x: (x/255.0) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

model.fit_generator(train_generator,samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=3)

model.save('model.h5')
exit()
