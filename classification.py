# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:26:05 2021

@author: user
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,models,layers

(training_images,training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data()

"""
scale the value to 0 and 1.
"""
training_images,testing_images=training_images/255,testing_images/255 

### to specify the names 
class_names=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

### visualise the image data
for i  in range(16):
    plt.subplot(4,4,i+1) # with the size 4 *4 and i+1 is iterate and put all the things 1 by 1.
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary) 
    plt.xlabel(class_names[training_labels[i][0]]) ### get the training labels index and pass into  class name
    print(training_labels[i])
    print(training_labels[i][0])
plt.show()

### take twenty thousand for training
### take four thousand for testing to save times.



# training_images=training_images[:20000]
# training_labels=training_labels[:20000]
# testing_images=testing_images[:4000]
# testing_labels=testing_labels[:4000]

# model=models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten()) ##  flatten into 1D vector 
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels),verbose=2)

# ### evaluate the models.
# loss,accuracy=model.evaluate(testing_images,testing_labels)
# print(f"loss:{loss}")
# print(f"accuracy:{accuracy}")

### save the model
# model.save('image_classifier.model')

model=models.load_model('image_classifier.model')

img=cv2.imread('horse.jpg')
### convert to rgb
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
resize_img=cv2.resize(img,(32,32))
# img_grey=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# cv2.imshow("",resize_img)
# cv2.waitKey()
# cv2.destroyAllWindows()



### prediction
prediction=model.predict(np.array([resize_img])/255)
print(f"prediction list are {prediction}")
# rethrieve the   the index of highest value in the list.
index=np.argmax(prediction)
print(index)
print(f" the prediction of this image is {class_names[index]}")


