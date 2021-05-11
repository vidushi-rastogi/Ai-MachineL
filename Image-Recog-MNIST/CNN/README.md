# Convolutional Neural Network
#
#
Image processing for MNNIST dataset of numbers using CNN.
Resuming from standard algorithm approach.


## Approach 
#
#
#### 4. Reshaping of dataset
#
In CNN, we don't need to flatten the dataset to 1D array, instead we add one more dimension for color channel.
As we are using grayscale images so channel will be 1, if it was RGB the channel would be 3
```sh
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
```
#
#
#### 2. Normalisation of dataset and One-Hot-encoding

```sh
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
 
X_train = X_train/255
X_test = X_test/255
```

#### 3. Creation of Sequential Model Using LeNet Architecture

**Feature Learning**
```sh
  model.add(Conv2D(30, (5, 5), input_shape= (28, 28, 1), activation="relu"))    
  model.add(MaxPooling2D(pool_size= (2, 2)))     
  model.add(Conv2D(15, (3, 3), activation= "relu"))    
  model.add(MaxPooling2D(pool_size= (2, 2)))    
```
#
**Classfication**
```sh
  model.add(Flatten()) 
  model.add(Dense(505, activation= "relu"))     
  model.add(Dropout(0.5))    #dropout layer
  model.add(Dense(num_classes, activation= "softmax"))    
```
Dropout layer is added to prevent over fitting in the model, with half of the layers dropping out randomly in each round.
0 - no layer drops out
0.5 - half of the layers drop out
1 - all the layers drop out
#
#
#### 4. Training the model
The training data for the model is split into two as 10% validation data and rest as training data
This determines the accuracy rate of the model for classifying the data.
```sh
model.fit(X_train, y_train, validation_split= 0.1, epochs= 10, batch_size= 400, verbose= 1, shuffle= 1)
```
Number of training rounds is 10 with batch size of 400
#

**Code and learning reference -**
https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/


