# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()

print("Training images shape",train_images.shape)
print("Test images shape",test_images.shape)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

train_images = train_images/255.0
test_images = test_images/255.0

train_labels = to_categorical(train_labels,num_classes = 10)
test_labels = to_categorical(test_labels,num_classes = 10)

"""![Dataset.png](attachment:Dataset.png)"""

for i in range(8):
  plt.imshow(train_images[i])
  plt.show()

model = Sequential()
model.add(Flatten(input_shape=[32,32,3]))
model.add(Dense(2048, activation='relu', input_dim = 1024))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'SGD', metrics=['accuracy'])

model.summary()

history = model.fit(train_images,train_labels,batch_size = 32,epochs = 10, callbacks = EarlyStopping(monitor = 'val_loss',patience= 2),validation_split= 0.1)

history = model.fit(train_images,train_labels,batch_size = 64,epochs = 10,callbacks = EarlyStopping(monitor = 'val_loss',patience= 2), validation_split= 0.1)

model2 = Sequential()
model2.add(Flatten(input_shape=[32,32,3]))
model2.add(Dense(2048, activation='relu', input_dim = 1024))
model2.add(BatchNormalization())
model2.add(Dense(1024, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(512, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(256, activation='relu'))
model2.add(Dense(10, activation='softmax'))

model2.build(input_shape=(None,32,32,3))
model2.compile(loss='categorical_crossentropy', optimizer= 'SGD', metrics=['accuracy'])

model2.fit(train_images,train_labels,batch_size = 32,epochs = 10, callbacks = EarlyStopping(monitor = 'val_loss',patience= 2),validation_split= 0.1)

cnmodel = Sequential()
cnmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnmodel.add(Conv2D(32, (3, 3), activation='relu'))
cnmodel.add(MaxPooling2D((2, 2)))
cnmodel.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel.add(MaxPooling2D((2, 2)))
cnmodel.add(Flatten(input_shape=[32,32,3]))
cnmodel.add(Dense(512, activation='relu',input_dim = 1024))
cnmodel.add(Dense(10, activation='softmax'))

cnmodel.summary()

cnmodel.compile(loss='categorical_crossentropy', optimizer= 'SGD', metrics=['accuracy'])
history = cnmodel.fit(train_images, train_labels,batch_size = 32, epochs=10, validation_split=0.1)

history = cnmodel.fit(train_images, train_labels,batch_size = 64, epochs=10, validation_split=0.1)

history = cnmodel.fit(train_images, train_labels,batch_size = 128, epochs=10, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')

cnmodel2 = Sequential()
cnmodel2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnmodel2.add(Conv2D(32, (3, 3), activation='relu'))
cnmodel2.add(Dropout(0.25))
cnmodel2.add(MaxPooling2D((2, 2)))
cnmodel2.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel2.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel2.add(Dropout(0.25))
cnmodel2.add(MaxPooling2D((2, 2)))
cnmodel2.add(Flatten(input_shape=[32,32,3]))
cnmodel2.add(Dense(512, activation='relu',input_dim = 1024))
cnmodel2.add(Dropout(0.25))
cnmodel2.add(Dense(10, activation='softmax'))

cnmodel2.compile(loss='categorical_crossentropy', optimizer= 'SGD', metrics=['accuracy'])
history = cnmodel2.fit(train_images, train_labels,batch_size = 32, epochs=10, validation_split=0.1)

cnmodel3 = Sequential()
cnmodel3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnmodel3.add(Conv2D(32, (3, 3), activation='relu'))
cnmodel3.add(Dropout(0.25))
cnmodel3.add(BatchNormalization())
cnmodel3.add(MaxPooling2D((2, 2)))
cnmodel3.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel3.add(Conv2D(64, (3, 3), activation='relu',))
cnmodel3.add(Dropout(0.25))
cnmodel3.add(MaxPooling2D((2, 2)))
cnmodel3.add(Flatten(input_shape=[32,32,3]))
cnmodel3.add(Dense(512, activation='relu',input_dim = 1024))
cnmodel3.add(Dropout(0.25))
cnmodel3.add(BatchNormalization())
cnmodel3.add(Dense(10, activation='softmax'))

cnmodel3.compile(loss='categorical_crossentropy', optimizer= 'SGD', metrics=['accuracy'])
history = cnmodel3.fit(train_images, train_labels,batch_size = 32, epochs=10, validation_split=0.1)

pre = cnmodel.predict(test_images)
pre = np.argmax(pre, axis=1)

# Converting the test labels to class labels
test_label_class = np.argmax(test_labels, axis=1)

# Making predictions on a few example images from the test set
predictions = model.predict(test_images[:5])
for i in range(5):
    
    probs = predictions[i]
    label = np.argmax(probs)
    plt.imshow(test_images[i])
    plt.title(f'Predicted: {label}')
    plt.xlabel('Class Probabilities')
    plt.xticks(range(10), labels=range(10))
    plt.yticks(range(10), labels=range(10))
    plt.barh(range(10), width=probs)
    plt.show()

matrix = confusion_matrix(test_label_class, pre)
print(matrix)
plt.imshow(matrix)

from tensorflow.keras.applications import ResNet50, VGG16
resmodel = ResNet50(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
vggmodel = VGG16(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)

predictions = resmodel.predict(test_images[:2])
for i in range(2):
    
    probs = predictions[i]
    label = np.argmax(probs)
    plt.imshow(test_images[i])
    plt.title(f'Predicted: {label}')
    plt.xlabel('Class Probabilities')
    plt.xticks(range(10), labels=range(10))
    plt.yticks(range(10), labels=range(10))
    plt.barh(range(10), width=probs)
    plt.show()

predictions = vggmodel.predict(test_images[:2])
for i in range(2):
    
    probs = predictions[i]
    label = np.argmax(probs)
    plt.imshow(test_images[i])
    plt.title(f'Predicted: {label}')
    plt.xlabel('Class Probabilities')
    plt.xticks(range(10), labels=range(10))
    plt.yticks(range(10), labels=range(10))
    plt.barh(range(10), width=probs)
    plt.show()
