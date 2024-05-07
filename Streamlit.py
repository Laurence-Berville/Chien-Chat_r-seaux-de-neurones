import streamlit as st
from main import evaluate3
from main2 import  evaluate2,evaluate1
import urllib.request as req
from keras.models import load_model  # Importation manquante

def accueil():
    st.header("Nos modèles de prédiction : ")
    st.image('./datafrog_chien_chat-2.jpg', width=350)
    st.markdown("---")
    st.write("- Réseau de neurones convolutionnel profond | learning rate 0,001 : Modèle 25 Epochs, drop out 0,9.")
    # Création d'une disposition en colonnes
    col1, col2 = st.columns(2)
    # Affichage de chaque image dans une colonne
    with col1:
        st.image('./plot25epoch.png', width=350)
    with col2:
        st.image('./plotloss25epoch.png', width=350)
    st.write("1. la précision d'entraînement ne s'approche pas de 100%, à cause du fort taux de dropout.")
    st.write("2. le réseau commence à être surentraîné dès l'époque 6.")
    st.write("3. la précision de validation sature autour de 88%.")

    st.markdown("---")
    st.write("- Réseau de neurones convolutionnel profond | learning rate 0.001 : Modèle 25 Epochs, drop out 0,4.")
    st.write("L'augmentation des données consiste à créer de nouveaux exemples pour l'entraînement à partir de ceux dont nous disposons déjà grâce à l'ImageDataGenerator. ImageDataGenerator va renverser à gauche ou droite, zoomer, et retourner les images, de façon aléatoire.")
 
    # Création d'une disposition en colonnes
    col1, col2 = st.columns(2)
    # Affichage de chaque image dans une colonne
    with col1:
        st.image('./plot25epoch_augm.png', width=350)
    with col2:
        st.image('./plotloss25epoch_augm.png', width=350)
    st.write("")
    st.write("1. le surentraînement est fortement réduit.")
    st.write("2. la précision de validation sature autour de 91%.")

    st.markdown("---")
    st.write("- Modèle pré-entraîné : ResNet50")
    st.write("> un modèle entraîné sur l'échantillon ImageNet , qui contient 14 millions d'images dans 1000 catégories.") 



def chargement_image():
    st.subheader("Chargement de l'image et restitution des prédictions")
    st.write("Pour nos 3 modèles.")
    st.markdown("---")
    
    # Affichage du multiselect pour choisir les modèles
    selected_models = st.multiselect('Choisissez les modèles :', ["Modèle 1", "Modèle 2", "Modèle 3"])
    
    # Code pour charger et afficher l'image
    image_file = st.file_uploader("Uploader une image", type=["jpg", "png"])
    
    if image_file is not None:
        st.image(image_file, caption="Image uploadée", use_column_width=True)
        
        # Évaluation de l'image avec les modèles sélectionnés
        for model_name in selected_models:
            if model_name == "Modèle 1":
                predictions = evaluate1(image_file)
            elif model_name == "Modèle 2":
                predictions = evaluate2(image_file)
            elif model_name == "Modèle 3":
                predictions = evaluate3(image_file)
                
            # Affichage des prédictions pour le modèle actuel
            st.write(f"Prédictions pour {model_name} :")
            for prediction in predictions:
                st.write(prediction)

def Images():
    st.subheader("Images du data set")
    st.write("Voici les images qui ont été enlevées du train set.")
    st.markdown("---")
       # Création d'une disposition en colonnes
    col1, col2 = st.columns(2)
    # Affichage de chaque image dans une colonne
    with col1:
        st.image('./M_chat.png', width=350)
    with col2:
        st.image('./M_chien.png', width=350)
    st.write("")
    st.write("1. Ce dataset a originellement été introduit pour une compétition Kaggle en 2013.")
    st.write("2. 12 500 chats & 12 500 chiens")
    st.write("3. Images enlevées : 21 chats & 22 chiens")

def explication_cnn():
    st.title("Explication pédagogique des réseaux de neurones CNN")
    st.markdown("""
Un CNN est un type de réseau de neurones utilisé principalement pour la vision par ordinateur.
Il est composé de couches spéciales :
    
1. **Convolution** : Application de filtres pour extraire des caractéristiques de l'image.
2. **Activation** : Intégration de non-linéarités pour apprendre des relations complexes.
3. **Pooling** : Réduction de la dimension spatiale pour la gestion efficace des informations.
4. **Couches entièrement connectées** : Combinaison des caractéristiques pour la classification finale.

#### Pourquoi les CNN sont-ils efficaces ?

- **Partage de poids** : Les filtres de convolution sont partagés, permettant une détection invariante à la translation.
- **Invariance spatiale** : Les opérations de pooling fournissent une invariance aux petites variations de position.
- **Hiérarchie de caractéristiques** : Les couches apprennent des caractéristiques de complexité croissante.
                
Forces :

Extraction hiérarchique des caractéristiques : Les CNN apprennent à extraire automatiquement des caractéristiques pertinentes à différents niveaux d'abstraction, ce qui leur permet de capturer des informations complexes.
Invariance spatiale : Les opérations de convolution et de pooling permettent aux CNN de détecter des motifs indépendamment de leur position dans l'image.
Performances élevées : Les CNN ont démontré des performances exceptionnelles dans un large éventail de tâches de vision par ordinateur, y compris la classification d'images, la détection d'objets et la segmentation sémantique.

Faiblesses :

Exigences en matière de données et de calcul : Les CNN nécessitent souvent de grandes quantités de données étiquetées pour être entraînés efficacement, ainsi que des ressources de calcul importantes pour l'entraînement et l'inférence.
Interprétabilité limitée : En raison de leur complexité, il peut être difficile d'interpréter les décisions prises par les CNN, ce qui peut poser des problèmes dans certains domaines sensibles où la transparence est importante.

Cas d'usage :

Classification d'images : Les CNN sont largement utilisés pour classer les images dans différentes catégories telles que les animaux, les objets, les scènes, etc.
Détection d'objets : Les CNN peuvent détecter et localiser la présence d'objets spécifiques dans une image, ce qui est utile dans les systèmes de surveillance, la robotique, etc.
Reconnaissance faciale : Les CNN sont utilisés pour identifier et vérifier l'identité des individus dans les systèmes de sécurité, les applications mobiles, etc.

En résumé, les CNN sont des modèles puissants pour la reconnaissance visuelle, offrant une combinaison de performances élevées, d'efficacité dans l'extraction des caractéristiques 
et de capacités d'apprentissage hiérarchique. Ils sont largement utilisés dans de nombreux domaines et continuent de jouer un rôle important dans les avancées de l'intelligence artificielle.
    """)
    st.markdown("---")





def main():
    st.sidebar.title("Chat ou chien ?")
    choix_onglet = st.sidebar.radio("Sélectionnez un onglet :", ("Accueil", "Chargement d'image", "Images","Explication CNN"))

    if choix_onglet == "Accueil":
        st.markdown(
            """
            <style>
            .reportview-container {
                background: #fcf7f7;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        accueil()
    elif choix_onglet == "Chargement d'image":
        st.markdown(
            """
            <style>
            .reportview-container {
                background:#fcf7f7;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        chargement_image()
    elif choix_onglet == "Images":
        st.markdown(
            """
            <style>
            .reportview-container {
                background:#fcf7f7;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        Images()

    elif choix_onglet == "Explication CNN":
        st.markdown(
            """
            <style>
            .reportview-container {
                background: #fcf7f7;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        explication_cnn()

if __name__ == "__main__":
    main()
