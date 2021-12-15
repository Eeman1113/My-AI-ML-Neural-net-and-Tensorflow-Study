# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!


#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
#print('Test accuracy:', test_acc)

#prediction part__________________________________________________________________________________________________
#predictions = model.predict(test_images)




#COLOR = 'white'
#plt.rcParams['text.color'] = COLOR
#plt.rcParams['axes.labelcolor'] = COLOR
#
#def predict(model, image, correct_label):
#  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#  prediction = model.predict(np.array([image]))
#  predicted_class = class_names[np.argmax(prediction)]
#
#  show_image(image, class_names[correct_label], predicted_class)
#
#
#def show_image(img, label, guess):
#  plt.figure()
#  plt.imshow(img, cmap=plt.cm.binary)
#  plt.title("Excpected: " + label)
#  plt.xlabel("Guess: " + guess)
#  plt.colorbar()
#  plt.grid(False)
#  plt.show()
#
#
#def get_number():
#  while True:
#    num = input("Pick a number: ")
#    if num.isdigit():
#      num = int(num)
#      if 0 <= num <= 1000:
#        return int(num)
#    else:
#      print("Try again...")
#
#num = get_number()
#image = test_images[num]
#label = test_labels[num]
#predict(model, image, label)