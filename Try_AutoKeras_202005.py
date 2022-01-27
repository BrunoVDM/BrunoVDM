# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#Source:
#https://autokeras.com/tutorial/image_classification/

import autokeras as ak
from tensorflow.keras.datasets import mnist
# -

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=1, overwrite=True) # It tries 1 different model.
# Feed the image classifier with training data.
clf.fit(x_train, y_train,epochs=3)

# Evaluate on the testing data.
print('Accuracy: {accuracy}'.format(accuracy=clf.evaluate(x_test, y_test)))

clf.evaluate(x_test, y_test)


