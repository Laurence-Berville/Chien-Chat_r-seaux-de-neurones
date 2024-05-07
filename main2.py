# import des packages nécessaires
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras
import glob
import re
import numpy as np
import shutil
from keras.models import load_model  # Import de la fonction load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# raccourci vers la classe ImageDataGenerator 
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator



model1 = load_model("model25_epoch.h5")
print(model1)
def evaluate1(img_fname):
    # Charger l'image avec les bonnes dimensions
    img = image.load_img(img_fname, target_size=(256, 256))
    # Convertir l'image en tableau numpy et normaliser les valeurs de pixel
    x = image.img_to_array(img) / 255.0
    # Ajouter une dimension supplémentaire pour correspondre à la forme attendue par le modèle
    x = np.expand_dims(x, axis=0)
    # Faire une prédiction avec le modèle
    preds1 = model1.predict(x)
    # Afficher les probabilités et les noms de catégorie pour les 2 catégories
    print('Prédictions:', preds1)
    # Calculer les pourcentages de prédiction
    percent_cat = preds1[0][1] * 100
    percent_dog = preds1[0][0] * 100
    # Formater les pourcentages de prédiction
    percent_cat_str = "Cat :{:.2f}".format(percent_cat)
    percent_dog_str = "Dog :{:.2f}".format(percent_dog)
    return percent_cat_str, percent_dog_str

model2 = load_model("model04_augm25_epoch.h5")
print(model2)
def evaluate2(img_fname):
    # Charger l'image avec les bonnes dimensions
    img = image.load_img(img_fname, target_size=(256, 256))
    # Convertir l'image en tableau numpy et normaliser les valeurs de pixel
    x = image.img_to_array(img) / 255.0
    # Ajouter une dimension supplémentaire pour correspondre à la forme attendue par le modèle
    x = np.expand_dims(x, axis=0)
    # Faire une prédiction avec le modèle
    preds2 = model2.predict(x)
    # Afficher les probabilités et les noms de catégorie pour les 2 catégories
    print('Prédictions:', preds2)
    # Calculer les pourcentages de prédiction
    percent_cat2 = preds2[0][1] * 100
    percent_dog2 = preds2[0][0] * 100
    # Formater les pourcentages de prédiction
    percent_cat_str2 = "Cat :{:.2f}".format(percent_cat2)
    percent_dog_str2 = "Dog :{:.2f}".format(percent_dog2)
    return percent_cat_str2, percent_dog_str2
