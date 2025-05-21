.. _guide_inference:

Effectuer une Inférence avec le Modèle Pré-entraîné
====================================================

Ce guide vous expliquera comment utiliser le notebook ``projet-vsr-model.ipynb`` pour charger le modèle de reconnaissance visuelle de la parole (VSR) pré-entraîné et effectuer des prédictions.

Prérequis
---------

1.  **Environnement Jupyter Notebook**:
    * Assurez-vous de disposer d'un environnement Python 3.11 fonctionnel avec Jupyter Notebook ou JupyterLab installé.
    * Vous aurez besoin du fichier notebook ``projet-vsr-model.ipynb``.

2.  **Bibliothèques Python Requises**:
    * Avant d'exécuter le notebook, installez les bibliothèques nécessaires. Les principales pour l'inférence sont :
        * ``torch``
        * ``numpy``
        * ``opencv-python`` (pour ``cv2``)
        * ``matplotlib`` (pour l'affichage optionnel des images)
        * ``ipykernel``
    * Vous pouvez généralement les installer via pip :
        .. code-block:: bash

           pip install torch numpy opencv-python matplotlib ipykernel pandas Sphinx sphinx-rtd-theme nbsphinx

3.  **Télécharger le Modèle Pré-entraîné**:
    * Téléchargez les poids du modèle pré-entraîné à partir du lien suivant :
        * **Lien de téléchargement du modèle**: ``[https://drive.google.com/file/d/1uH4XBYAxOgG9__LCA9VlcP5GCafqN1Uz/view?usp=sharing]``
    * Enregistrez le fichier du modèle téléchargé (par ex., ``model_fully_trained_16_05_09pm.pth``) dans un emplacement connu sur votre ordinateur. Vous aurez besoin de son chemin d'accès plus tard.

4.  **Télécharger les Données**:
    * Téléchargez les données d'exemple (séquence d'images) à partir de ce lien :
        * **Lien de téléchargement des données**: ``[https://drive.google.com/file/d/1tNfMa5MpWyVJte_91c1sq25l6nhGSsiN/view?usp=sharing]``

Exécution du Notebook pour l'Inférence
---------------------------------------

Ouvrez le notebook ``projet-vsr-model.ipynb`` et suivez ces étapes :

1.  **Cellule 1 : Définition de `CharMap`**
    * Exécutez cette cellule. Elle est indispensable pour le mappage des caractères.

2.  **Cellule 2 : Définition de `VSRDataset` et `collate_fn`**
    * Exécutez cette cellule. Elle définit comment les données sont chargées et préparées.

3.  **Cellule 3 : Définition de `NewVSRModel`**
    * Exécutez cette cellule. Elle définit l'architecture du réseau de neurones.

4.  **Cellule 4 : Configuration**
    * **Action Requise**: Modifiez le chemin d'accès aux données.
        * Localisez la ligne : ``KAGGLE_DATASET_ROOT_DIR = "/kaggle/input/30fpsdata/DATA"``
        * Remplacez ``"/kaggle/input/30fpsdata/DATA"`` par le chemin complet où vous avez placé le dossier ``DATA`` téléchargé à l'étape "Prérequis 3".
            *Exemple*: ``KAGGLE_DATASET_ROOT_DIR = "C:/Utilisateurs/VotreNom/Documents/ProjetVSR/DATA"``
    
* Exécutez la Cellule 4 après ces modifications.

5.  **Cellule 5 : Initialisation des Composants **
    * **Action Requise**: Cette cellule initialise ``train_dataset`` et ``train_loader``, ce qui est nécessaire pour que la Cellule 8 puisse y puiser un échantillon de données.
    * Le chargement du modèle pour l'inférence se fera dans la Cellule 8.
    * Exécutez cette cellule. Elle va créer ``train_dataset`` basé sur le ``KAGGLE_DATASET_ROOT_DIR`` que vous avez défini.

    .. warning::
       N'exécutez **PAS** la Cellule 6 et 7 (Boucle d'entraînement).

6.  **Cellule 8 : Chargement du Modèle Sauvegardé et Prédictions**
    * **Action Requise**: Modifiez le chemin du modèle à charger pour l'inférence.
        * Localisez la ligne : ``MODEL_TO_LOAD_PATH = "/kaggle/working/model_fully_trained_16_05_09pm.pth"``.
        * Remplacez cette chaîne de caractères par le chemin d'accès complet où vous avez sauvegardé le modèle pré-entraîné téléchargé (Prérequis 2).
            *Exemple*: ``MODEL_TO_LOAD_PATH = "C:/chemin/vers/mon_modele_telecharge.pth"``
    * **Vérification de l'index de l'échantillon (Optionnel)**:
        * La cellule utilise ``sample_idx_to_predict = 401`` pour choisir un échantillon du ``train_dataset``. Vous pouvez changer cet index (par exemple, entre 0 et ``len(train_dataset) - 1``) pour tester la prédiction sur différentes séquences du jeu de données que vous avez fourni.
    * Exécutez la Cellule 8.

7.  **Interprétation des Résultats (Cellule 8)**:
    * La sortie affichera le texte prédit pour l'échantillon choisi.
    * Vous pouvez modifier les paramètres de la fonction ``char_map_instance.indices_to_text()`` (``remove_blanks``, ``remove_duplicates``) dans la Cellule 8 pour observer la sortie brute du modèle si vous le souhaitez.

En résumé :
1. Téléchargez le modèle et les données.
2. Ouvrez le notebook.
3. Exécutez les cellules 1, 2, 3.
4. Modifiez ``KAGGLE_DATASET_ROOT_DIR`` dans la Cellule 4 et exécutez-la.
5. Exécutez la Cellule 5 pour charger les données.
6. Modifiez ``MODEL_TO_LOAD_PATH`` dans la Cellule 8 pour pointer vers votre modèle téléchargé, puis exécutez la Cellule 8.
