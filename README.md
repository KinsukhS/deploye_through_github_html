Traffic Sign Recognition Project - Experimentation Process README
=================================================================

Overview
--------

This document outlines the iterative process of developing a neural network for classifying road signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The goal is to enhance model performance by adjusting the architecture and observing the impact on loss and accuracy.

Initial Model Architecture
--------------------------

pythonCopy code

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES))`

Observations:

-   The initial model exhibited high loss, potentially due to large filter size and pooling size.
-   Struggled to capture essential features.

Adjustments in Try 2
--------------------

pythonCopy code

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES))`

Changes:

-   Reduced filter size and pooling size.

Observations:

-   Significant improvement; decreased loss from 10.4338 to 6.0791, with a slight increase in accuracy.

Architecture Enhancement in Try 3
---------------------------------

pythonCopy code

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))`

Changes:

-   Applied ReLU activation in Conv2D layers.
-   Added a second Conv2D layer with MaxPooling.

Observations:

-   Substantial increase in accuracy (0.9393) and decrease in loss (6.0791 to 0.6742).

Dropout Layer Experimentation in Try 4
--------------------------------------

pythonCopy code

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))`

Changes:

-   Added a Dropout layer after the first MaxPooling2D layer.

Observations:

1.  Dropout (0.4): Loss=0.2900, Accuracy=0.9159.
2.  Dropout (0.5): Loss=0.3456, Accuracy=0.9006.

Final Model Architecture in Try 5
---------------------------------

pythonCopy code

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))`

Summary:

-   The final architecture demonstrated robust performance with a significant increase in accuracy and a substantial reduction in loss.
-   Experimentation involved adjusting filter sizes, pooling sizes, activation functions, adding layers, and introducing dropout to enhance model robustness.
-   The final architecture is well-balanced, effectively capturing relevant features.

Background and Getting Started
------------------------------

As research progresses in self-driving cars, computer vision becomes crucial for understanding the environment. This project focuses on classifying road signs using TensorFlow and the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

### Getting Started Steps:

1.  Download the distribution code and data set.
2.  Move the GTSRB dataset into the project's data directory.
3.  Install dependencies using `pip3 install -r requirements.txt`.

Implementation Details
----------------------

### `load_data` Function

The `load_data` function accepts the `data_dir` argument, representing the path to the dataset. It returns image arrays and labels for each image. The function ensures platform independence by using `os.sep` and `os.path.join`. Images are resized using OpenCV-Python (cv2).

### `get_model` Function

The `get_model` function returns a compiled neural network model. It assumes input shape (IMG_WIDTH, IMG_HEIGHT, 3) and outputs NUM_CATEGORIES units for different traffic sign categories. Experimentation involves adjusting the number of layers, filter sizes, pooling sizes, hidden layers, and dropout.

Conclusion
----------

The project involved exploring various options in cv2 and TensorFlow, adjusting the neural network architecture, and observing the effects on model performance. The final model achieved a balance between accuracy and loss, effectively classifying road signs in the GTSRB dataset.
