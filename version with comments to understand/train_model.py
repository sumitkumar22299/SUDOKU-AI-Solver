import json
from random import shuffle

import cv2
import keras
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
# if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
batch_size = 64
num_classes = 9
epochs = 5

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
x_train = []
for i in range(8235):

    # Formatted string literals (also called f-strings for short) let you 
    # include the value of Python expressions inside a string by prefixing the string with f
    #  or F and writing expressions as {expression}.

    # EXAMPLE
    # print(f'The value of pi is approximately {math.pi:.3f}.')
    # The value of pi is approximately 3.142.

    fp = f'chars_train/{i}.png'


    # cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. 
    #                   It is the default flag. Alternatively, we can pass integer value 1 for this flag.
    # cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. 
    #                       Alternatively, we can pass integer value 0 for this flag.

    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 231, 0)

    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)

    # INTER_LANCZOS4 = Lanczos interpolation over 8x8 neighborhood , MORE COMPLEX , BUT SLOWER , 
    # works best for in both cases in enlarging and shrinking images!!!!! as we dont know what needs to be done 

    x_train.append(img)

x_test = []
for i in range(909):
    fp = f'chars_test/{i}.png'
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 231, 0)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
    x_test.append(img)

with open('labels_train.json') as f:
    y_train = json.load(f)
    y_train = [i - 1 for i in y_train]

with open('labels_test.json') as f:
    y_test = json.load(f)
    y_test = [i - 1 for i in y_test]

# NumPy is used to work with arrays. The array object in NumPy is called ndarray.

# We can create a NumPy ndarray object by using the array() function.
y_train, y_test, x_train, x_test = np.array(y_train), np.array(y_test), np.array(x_train), np.array(x_test)

# x is a 2D array, which can also be looked upon as an array of 1D arrays, having 10 rows and 1024 columns. 
# x[0] is the first 1D sub-array which has 1024 elements (there are 10 such 1D sub-arrays in x), and 
# x[0].shape gives the shape of that sub-array, which happens to be a 1-tuple - (1024, ).

# On the other hand, x.shape is a 2-tuple which represents the shape of x, which in this case is (10, 1024). 
# x.shape[0] gives the first element in that tuple, which is 10.
# x[0].shape will give the Length of 1st row of an array.
#  x.shape[0] will give the number of rows in an array. 
ind_list_train = [i for i in range(x_train.shape[0])]
shuffle(ind_list_train)
x_train = x_train[ind_list_train, :, :, ]     
# WTFFFFFFFF
y_train = y_train[ind_list_train,]

ind_list_test = [i for i in range(x_test.shape[0])]
shuffle(ind_list_test)
x_test = x_test[ind_list_test, :, :, ]
y_test = y_test[ind_list_test,]


#  Putting data_format to channel_first, you say that for every layer your tensor will have 
#  this shape: (batch, channels, height, width), 
#  but for channel_last you gonna have (batch, height, width, channels).


if K.image_data_format() == 'channels_first':
    # we want to transform our dataset form having shape (n,width,height) to (n,depth,width,height)

    # When using the Theano backend, you must explicitly declare a dimension for the depth of the input image.
    #  For example, a full-color image with all 3 RGB channels will have a depth of 3.
    # Our MNIST images only have a depth of 1, but we must explicitly declare that.

    # In other words, we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height).

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# The final preprocessing step for input data is to convert our datatype to  float32 and normalize our 
# data values to range [0,1]

# It is most common to use 32-bit precision when training a neural network, 
# so at one point the training data will have to be converted to 32 bit floats. 
# Since the dataset fits easily in RAM, we might as well convert to float immediately.

# Regarding the division by 255, this is the maximum value of a byte 
# (the input feature's type before the conversion to float32), 
# so this will ensure that the input features are scaled between 0.0 and 1.0. 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# print X_train.shape
# # (60000, 28, 28)
# Great, so it appears that we have 60,000 samples in our training set, and the images are 28 pixels x 28 pixels each.

print(x_test.shape[0], 'test samples')

# Syntax: tf.keras.utils.to_categorical(y, num_classes=None, dtype=”float32″)

# Parameters: 

# y (input vector):     A vector which has integers representing various classes in the data.

# num_classes:    Total number of classes. If nothing is mentioned, it considers the largest number of the input vector and adds 1,
#                 to get the number of classes. Its default value is "None".

# dtype:             It is the desired data type of the output values. By default, it's value is 'float32'.
# Output: 
# This function returns a matrix of binary values (either ‘1’ or ‘0’). 
# It has number of rows equal to the length of the input vector and number of columns equal to the number of classes.


y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)



# A Sequential model is appropriate for a plain stack of layers where each layer has
#  exactly one input tensor and one output tensor.


# A tensor is a container which can house data in N dimensions. generally N >= 3 .
# Often and erroneously used interchangeably with the matrix (which is specifically a 2-dimensional tensor), 
# tensors are generalizations of matrices to N-dimensional space.
# A Sequential model is not appropriate when:

# Your model has multiple inputs or multiple outputs
# Any of your layers has multiple inputs or multiple outputs
# You need to do layer sharing
# You want non-linear topology (e.g. a residual connection, a multi-branch model)

model = Sequential()


# 2D convolution layer (e.g. spatial convolution over images).

# This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. 
# When using this layer as the first layer in a model, provide the keyword argument input_shape 
# (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 3) 
# for 128x128 RGB pictures in data_format="channels_last"


# A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of multiple kernels stacked together.
# In image processing kernel is a convolution matrix or masks which can be used for blurring,
#  sharpening, embossing, edge detection, and more by doing a convolution between a kernel and an image.


# kernel_initializer: Initializer for the kernel weights matrix (see keras.initializers). Defaults to 'glorot_uniform'.
# -----GLOROT UNIFORM
# Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))
#  (fan_in is the number of input units in the weight tensor and fan_out is the number of output units).






# --------- RELU RELU RELU relu function---------
# tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
# Applies the rectified linear unit activation function.
# With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.
# Modifying default parameters allows you to use non-zero thresholds, 
# change the max value of the activation, and to use a non-zero multiple of the input for values below the threshold.

# For example:

# >>> foo = tf.constant([-10, -5, 0.0, 5, 10], dtype = tf.float32)
# >>> tf.keras.activations.relu(foo).numpy()
# array([ 0.,  0.,  0.,  5., 10.], dtype=float32)
# >>> tf.keras.activations.relu(foo, alpha=0.5).numpy()
# array([-5. , -2.5,  0. ,  5. , 10. ], dtype=float32)
# >>> tf.keras.activations.relu(foo, max_value=5).numpy()
# array([0., 0., 0., 5., 5.], dtype=float32)
# >>> tf.keras.activations.relu(foo, threshold=5).numpy()
# array([-0., -0.,  0.,  0., 10.], dtype=float32)

# Arguments------------

# x: Input tensor or variable.
# alpha: A float that governs the slope for values lower than the threshold.
# max_value: A float that sets the saturation threshold (the largest value the function will return).
# threshold: A float giving the threshold value of the activation function below which values will be damped or set to zero.
# Returns

# A Tensor representing the input tensor, transformed by the relu activation function. 
# Tensor will be of the same shape and dtype of input x.

# # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
# As far as choosing the appropriate value for no. of filters, it is always recommended to use powers of 2 as the values.


# 32 = (i.e. the number of output filters in the convolution).
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))


model.add(Conv2D(64, (3, 3), activation='relu'))


# ----------Max pooling operation for 2D spatial data.

# Downsamples the input along its spatial dimensions (height and width) by taking the maximum value 
# over an input window (of size defined by pool_size) for each channel of the input. 
# The window is shifted by strides along each dimension.

# pool_size: integer or tuple of 2 integers, window size over which to take the maximum. 
# (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified,
#  the same window length will be used for both dimensions.

# strides: Integer, tuple of 2 integers, or None. Strides values. 
# Specifies how far the pooling window moves for each pooling step. If None, it will default to pool_size.


model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout works by randomly setting the outgoing edges of hidden units 
# (neurons that make up hidden layers) to 0 at each update of the training phase.
#  Dropout is a technique used to prevent a model from overfitting.

# without dropout, the validation loss stops decreasing after the third epoch.
# without dropout, the validation accuracy tends to plateau around the third epoch.

# In passing 0.5, every hidden unit (neuron) is set to 0 with a probability of 0.5. 
# In other words, there’s a 50% change that the output of a given neuron will be forced to 0.
model.add(Dropout(0.25))

# Flatten is used to flatten the input. For example, if flatten is applied to layer having input shape as (batch_size, 2,2), 
# then the output shape of the layer will be (batch_size, 4)

model.add(Flatten())

# What is a dense neural network?
# The name suggests that layers are fully connected (dense) by the neurons in a network layer.
#  Each neuron in a layer receives an input from all the neurons present in the previous layer—thus, they’re densely connected.

# units: Positive integer, dimensionality of the output space.

# Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is 
# the element-wise activation function passed as the activation argument, kernel is a weights matrix created 
# by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
#  These are all attributes of Dense.

# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
# ZEROS= Initializer that generates tensors initialized to 0.

model.add(Dense(12544, activation='relu'))

model.add(Dropout(0.5))


# Softmax converts a vector of values to a probability distribution.
# The elements of the output vector are in range (0, 1) and sum to 1.
# Each vector is handled independently. The axis argument sets which axis of the input the function is applied along.
# Softmax is often used as the activation for the last layer of a classification network
#  because the result could be interpreted as a probability distribution.


model.add(Dense(num_classes, activation='softmax'))

-----------------------------------------
# compile method
# Configures the model for training.

# optimizer: String (name of optimizer) or optimizer instance.
# metrics: List of metrics to be evaluated by the model during training and testing.



# Categorical crossentropy is a loss function that is used in multi-class classification tasks. 
# These are tasks where an example can only belong to one out of many possible categories, and the model must decide which one.
# This loss is a very good measure of how distinguishable two discrete probability distributions are from each other.

# The MNIST number recognition tutorial, where you have images of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.
# The model uses the categorical crossentropy to learn to give a high probability to the correct digit and 
# a low probability to the other digits.

# Also called Softmax Loss. It is a Softmax activation plus a Cross-Entropy loss.
#  one could think of cross-entropy as the distance between two probability distributions in
#   terms of the amount of information (bits) needed to explain that distance. 
#   It is a neat way of defining a loss which goes down as the probability vectors get closer to one another.
# Use a single Categorical feature as target.
# This will automatically create a one-hot vector from all the categories identified in the dataset. 
# Each one-hot vector can be thought of as a probability distribution, which 
# is why by learning to predict it, the model will output a probability that an example belongs to any of the categories.

# --------------------------------------
# Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
#  window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
#   continues learning even when many updates have been done. Compared to Adagrad, in 
#   the original version of Adadelta you don't have to set an initial learning rate. 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary() 
# ----------fit method-------------------
# Trains the model for a fixed number of epochs (iterations on a dataset).
# x= input data y= output data 
# verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
# Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
# OUTPUT+++@@@@@      A History object. Its History.history attribute is a record of training loss values and 
# metrics values at successive epochs, as well as validation loss values and 
# validation metrics values (if applicable).

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# _________EVALUTE_________
# Returns the loss value & metrics values for the model in test mode.

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model2.hdf5')
