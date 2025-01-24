# Image-Classification-Project
model to classify the images into one of six categories: Bear, Camel, Chicken, Elephant, Horse, Lion, or Squirrel 
# Image Classification Project

## Description
Ce projet utilise un modèle CNN pour classer les images dans les catégories suivantes :
- Bear
- Camel
- Chicken
- Elephant
- Horse
- Lion
- Squirrel

## Structure du projet
Image-Classification-Project/ │ ├── data/ # Dossier contenant les images train et test ├── src/ # Scripts Python pour traitement et entraînement ├── models/ # Modèle entraîné sauvegardé ├── results/ # Graphiques des résultats ├── notebooks/ # Notebook exploratoire ├── README.md # Documentation du projet ├── requirements.txt # Dépendances nécessaires └── .gitignore #


## Installation
1. Cloner le dépôt :
    ```
    git clone https://github.com/mohamediaaraben/Image-Classification-Project.git
    cd Image-Classification-Project
    ```

2. Installer les dépendances :
    ```
    pip install -r requirements.txt
    ```

3. Exécuter l'entraînement :
    ```
    python src/train_model.py
    ```

4. Évaluer le modèle :
    ```
    python src/evaluate_model.py
    ```

## Améliorations futures
- Ajouter des couches de dropout pour éviter l'overfitting.
- Expérimenter avec des modèles plus complexes comme ResNet.
- Augmenter les données via data augmentation.

## Auteur
Mohamed Iaaraben  
Email: mohamed.iaaraben@etu.uae.ac.ma  
GitHub: VOTRE_USERNAME

