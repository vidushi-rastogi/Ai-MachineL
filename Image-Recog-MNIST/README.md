# Image Recognition
#
#
Image processing for MNNIST dataset of numbers using Machine Learning. This is done through using two different neural networks

- Artifial Neural Network (ANN)
- Convolutional Neural Network (CNN)


## Approach for the standard algorithm
#
#
#### 1. Importing MNINST dataset
#
Dividing test dataset and training dataset
```sh
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
```

Also, assuring the imported dataset is matched with model requirements and valid using `assert` command
```sh
assert X_train.shape[0] == y_train.shape[0], "number of image is not equal to the number of labels"
assert X_test.shape[0] == y_test.shape[0], "number of image is not equal to the number of labels"
assert X_train.shape[1:] == (28,28), "dimension of the images are not 28x28"
assert X_test.shape[1:] == (28,28), "dimension of the images are not 28x28" 
```
#
#
#### 2. Displaying sample data and mapping it to grayscale channel
#
In the code only 5 samples of each number (0-9) are displayed. A matrix of 10x5 is plotted using `subplot` from pyplot. With iteration each image is converted to grayscale channel
#
```sh
fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5, 10))
fig.tight_layout()
```
```sh
axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap = plt.get_cmap("gray"))
```
#
Sample Output
![alt text](https://www.researchgate.net/profile/Alessandro-Di-Nuovo/publication/328030580/figure/fig1/AS:677340703121411@1538502016731/Examples-handwritten-digits-in-the-MNIST-dataset.ppm)
#
> Note: This output image is just to show what the code does, the actual output will be different.
#
#
#### 3. Distribution of Training dataset
#
This is done to visualise the distribution of dataset among all 10 numbers. This is done using `pyplot`
#
```sh
print(num_of_samples)
plt.figure(figsize= (12, 4))
plt.bar(range(0, num_classes), num_of_samples)  #plots bar graph
plt.title("Distribution of training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
```
#
#
#
> Note: The above mentioned steps are common in both approaches (ANN or CNN)
#
#
#
**Code and learning reference -**
https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/


