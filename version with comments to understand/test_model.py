import cv2
import keras
import numpy as np
from keras.models import load_model

model = load_model('model1.hdf5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

img_rows, img_cols = 32, 32

for i in range(81):
    fp = f'chars_test/{i}.png'
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    # img = cv2.fastNlMeansDenoising(img, None, 9, 13)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # img = cv2.bitwise_not(img)
    img = cv2.resize(img, (img_rows, img_cols), cv2.INTER_LANCZOS4)
    arr = np.array([img])
    reshaped = arr.reshape(arr.shape[0], img_rows, img_cols, 1)
    flt = reshaped.astype('float32')
    flt /= 255

#  predict will return the scores of the regression and predict_class will return the class of your prediction. 
# Although it seems similar there are some differences:
# Imagine you are trying to predict if the picture is a dog or a cat (you have a classifier):
# predict will return you: 0.6 cat and 0.4 dog (for example).
# predict_class will return you cat
# Now image you are trying to predict house prices (you have a regressor):
# predict will return the predicted price
# predict_class will not make sense here since you do not have a classifier
# TL:DR: use predict_class for classifiers (outputs are labels) and use predict for regressions (outputs are non discrete)

    classes = model.predict_classes(flt)
    print(f'{i}.png: {classes[0] + 1}')
    cv2.imshow('x', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
