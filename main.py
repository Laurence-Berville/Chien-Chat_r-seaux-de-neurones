
# import des packages nécessaires
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras
import glob
import re
import numpy as np
import shutil
import urllib.request as req
import os
from keras.models import load_model  # Importation manquante
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# raccourci vers la classe ImageDataGenerator 
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import preprocess_input, decode_predictions


model3 = ResNet50(weights='imagenet')
def evaluate3(img_fname):
    img = image.load_img(img_fname, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds3 = model3.predict(x)
    predictions3 = decode_predictions(preds3, top=3)[0]
    return predictions3




