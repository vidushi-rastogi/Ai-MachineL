# Traffic Signs Image Processing
#
#
Processing Traffic signs images using Convolutional Neural Network using a *Modified LeNet Model*
Data set for training the model used from - https://bitbucket.org/jadslim/german-traffic-signs
Approach is very much same as in MNIST Image processing using CNN (https://github.com/Ackermann99/Ai-MachineL/tree/master/Image-Recog-MNIST/CNN)
Some more modification is added to improve the data set by adding a variety of augmentation.

## Augmentation
From python image processing library importing `ImageDataGenerator`, and using it to add variations in the images.
```
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
```
#
#
## Training the model with augmented data
#
```
batch_size = 50
steps_per_epoch = X_train.shape[0]/batch_size
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=10, validation_data=(X_val, y_val), shuffle=1)
```

>Note: Training the model multiple times can improve its accuracy, also manipulating some parameters can lead to a better trained model.

#
#
**Code and learning reference -**
https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/


