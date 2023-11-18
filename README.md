Traffic Sign Classification Experimentation
-------------------------------------------

### Introduction

In this project, the goal was to build an effective Convolutional Neural Network (CNN) for the classification of traffic signs using TensorFlow and OpenCV. The experimentation process involved adjusting various aspects of the model architecture and hyperparameters to observe their impact on performance metrics.

### Experimentation Process

#### Initial Model (Model 1)

The first model incorporated a convolutional layer with a large filter size and max-pooling with a 3x3 pool size. However, this initial architecture resulted in high loss, prompting the need for adjustments.

#### Iterative Refinement (Models 2-3)

In subsequent iterations, I experimented with reducing the filter and pool sizes. Model 2, with a smaller filter and pool size, showed a decrease in loss and a modest improvement in accuracy. To enhance the model further, I introduced activation functions and a dropout layer in Model 3, which significantly improved accuracy while reducing loss.

#### Further Adjustments and Observations

I conducted additional experiments by adding an extra Conv2D and MaxPooling layer, adjusting dropout rates, and varying the number of filters in Conv2D layers. Notably, a dropout layer after the flatten layer resulted in a significant accuracy boost.

#### Final Model (Model 4)

The conclusive model (Model 4) utilized a balanced architecture with a smaller filter and pool size, activation functions, dropout layers, and a suitable number of filters in Conv2D layers. This configuration led to a highly accurate and well-generalizing model.

Initial Model Architecture
--------------------------

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

-   Dropout (0.4): Loss=0.2900, Accuracy=0.9159.
-   Dropout (0.5): Loss=0.3456, Accuracy=0.9006.

Final Model Architecture in Try 5
---------------------------------

`model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))`

---

### What Worked Well

-   Reducing filter and pool sizes improved model performance.
-   Activation functions and dropout layers enhanced the model's ability to generalize.
-   Experimenting with different architectures led to a final model with high accuracy.

### What Didn't Work Well

-   Larger filter and pool sizes in the initial model resulted in high loss.
-   Some dropout rates led to a decrease in accuracy, emphasizing the importance of finding an optimal rate.

### Key Observations

-   Architecture adjustments significantly impact model performance.
-   Iterative refinement is crucial for finding the optimal model configuration.
-   Dropout layers at strategic positions can enhance the model's ability to generalize.
