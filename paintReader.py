import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Model Upload
new_model_uploaded = tf.keras.models.load_model('shape.model')

new_model_uploaded.summary()
# Current Folder Directory
currentDirectory = os.getcwd()


# Convert Image to Gray Scale
firstRead = Image.open(currentDirectory+'\\test.png')
firstRead = np.array(firstRead)
firstRead = 255-firstRead

# Saving Image with Reversed Color Version
savedImage = Image.fromarray(firstRead)
savedImage.save("Tested.png")

# Reread Reversed Color Version
readImage = Image.open(currentDirectory+'\\Tested.png').convert('L')
readImage = np.array(readImage)
readImage = readImage.reshape(1, 28, 28)
readImage = tf.keras.utils.normalize(readImage, axis=1)
# print(readImage)  # Printing Read Image ---------------------------

plt.imshow(readImage.reshape(28, 28))  # Drawn DATA
plt.show()

predicting2 = new_model_uploaded.predict(readImage.reshape(1, 28, 28))  # Drawn DATA-Reshaped for 3D Matrix
predicting_value2 = np.argmax(predicting2)


# Printing Predicted Results
print('Drawn Value: ' + str(predicting_value2))
print('_________________________________________________________________')
