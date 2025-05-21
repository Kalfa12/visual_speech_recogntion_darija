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
        * **Lien de téléchargement du modèle**: ``[VOTRE LIEN DE TÉLÉCHARGEMENT DU MODÈLE ICI]``
    * Enregistrez le fichier du modèle téléchargé (par ex., ``model_fully_trained_16_05_09pm.pth``) dans un emplacement connu sur votre ordinateur. Vous aurez besoin de son chemin d'accès plus tard.

4.  **Télécharger les Données de Test**:
    * Téléchargez les données de test d'exemple (séquence d'images) à partir de ce lien :
        * **Lien de téléchargement des données de test**: ``[VOTRE LIEN DE TÉLÉCHARGEMENT DES DONNÉES DE TEST ICI]``
    * Extrayez les données dans un répertoire. Vous aurez besoin du chemin d'accès à ce répertoire.

Exécution du Notebook pour l'Inférence
---------------------------------------

Ouvrez le notebook ``projet-vsr-model.ipynb`` et procédez comme suit :

1.  **Exécuter les Cellules de Configuration Essentielles**:
    Pour que la Cellule 8 (cellule de prédiction) fonctionne correctement, vous devez exécuter les cellules qui définissent les classes et configurations nécessaires.
    * **Exécutez la Cellule 1**: Cette cellule définit la classe ``CharMap``.
    * **Exécutez la Cellule 3**: Cette cellule définit la classe ``NewVSRModel`` (l'architecture du modèle).
    * **Exécutez les Parties Pertinentes de la Cellule 4 (Configuration)**: Cette cellule configure des paramètres importants. Assurez-vous que les variables Python suivantes sont définies en exécutant les lignes correspondantes dans la Cellule 4 :
        * ``IMG_HEIGHT = 96``
        * ``IMG_WIDTH = 96``
        * ``GRU_HIDDEN_DIM = 256`` (ou la valeur correcte pour le modèle pré-entraîné)
        * ``GRU_NUM_LAYERS = 2`` (ou la valeur correcte)
        * ``BOTTLENECK_DIM = 256`` (ou la valeur correcte)
        * La variable ``arabic_alphabet_str``.
        * La variable ``device`` (CPU ou CUDA).

        .. important::
           Vous n'avez **pas** besoin d'exécuter les parties de la Cellule 4 relatives aux chemins du jeu de données pour l'entraînement (comme `KAGGLE_DATASET_ROOT_DIR`) ou aux hyperparamètres d'entraînement si vous effectuez uniquement une inférence. De plus, **n'exécutez pas la Cellule 5 (Initialize Components & Load Checkpoint pour l'entraînement) ni la Cellule 6 (Training Loop)**, car elles sont destinées à l'entraînement du modèle.

2.  **Modifier la Cellule 8 ("Load Saved Model and Making Predictions")**:
    C'est la cellule principale pour l'inférence. Avant de l'exécuter, vous **devez** mettre à jour les chemins suivants :
    * Localisez la ligne : ``MODEL_TO_LOAD_PATH = "/kaggle/working/model_fully_trained_16_05_09pm.pth"``
        * **Action**: Modifiez la chaîne de caractères du chemin pour indiquer l'emplacement où vous avez sauvegardé le fichier du modèle pré-entraîné que vous avez téléchargé.
            Par exemple : ``MODEL_TO_LOAD_PATH = "C:/Utilisateurs/VotreUser/Telechargements/model_fully_trained.pth"`` ou ``MODEL_TO_LOAD_PATH = "/home/youruser/models/model_fully_trained.pth"``

    * Par défaut, la Cellule 8 utilise un échantillon du `train_dataset` via `sample_idx_to_predict = 401` et `sample_data = train_dataset[sample_idx_to_predict]`.
        Pour utiliser vos propres données de test téléchargées, vous devrez adapter la partie chargement des données de la Cellule 8.

        **Adaptation Recommandée pour la Cellule 8 (après le chargement du modèle)**:
        Au lieu de la section qui charge les données depuis `train_dataset` :
        ```python
        # 3. Prepare Input Data (Example: from train_dataset)
        # if 'train_dataset' in globals() and train_dataset is not None and len(train_dataset) > 0:
        #     sample_idx_to_predict = 401 # Change as needed
        #     # ... (lignes pour charger depuis train_dataset) ...
        # else:
        #     print("  [Prediction] train_dataset not available/empty...")
        ```
        **Action**: Remplacez (ou commentez) ce bloc et insérez le code suivant pour charger vos images depuis un répertoire spécifié. Assurez-vous d'importer `glob` et `cv2` si ce n'est pas déjà fait en haut de la cellule.

        ```python
        # --- MODIFICATION POUR LA CELLULE 8 : Bloc de chargement des données externes ---
        import glob # Assurez-vous que glob est importé
        import cv2  # Assurez-vous que cv2 est importé (opencv-python)

        # MODIFIEZ CE CHEMIN pour pointer vers le répertoire contenant VOS images de test
        PATH_TO_YOUR_TEST_FRAMES_DIR = "CHEMIN/VERS/VOTRE/DOSSIER/IMAGES_DE_TEST" 
        
        frames_for_pred = []
        img_size_pred = (IMG_HEIGHT, IMG_WIDTH) # Utilise les variables définies précédemment
        
        # Utilise glob pour trouver les fichiers d'images (png, jpg, jpeg)
        frame_files_pred = sorted(glob.glob(os.path.join(PATH_TO_YOUR_TEST_FRAMES_DIR, '*.png')))
        frame_files_pred += sorted(glob.glob(os.path.join(PATH_TO_YOUR_TEST_FRAMES_DIR, '*.jpg')))
        frame_files_pred += sorted(glob.glob(os.path.join(PATH_TO_YOUR_TEST_FRAMES_DIR, '*.jpeg')))

        if not frame_files_pred:
            print(f"ERREUR : Aucune image trouvée dans {PATH_TO_YOUR_TEST_FRAMES_DIR}")
            sample_frames_tensor = None
        else:
            for frame_filepath_pred in frame_files_pred:
                img_bgr_pred = cv2.imread(frame_filepath_pred, cv2.IMREAD_COLOR)
                if img_bgr_pred is None: 
                    print(f"Attention : Impossible de lire l'image {frame_filepath_pred}")
                    continue
                img_resized_pred = cv2.resize(img_bgr_pred, img_size_pred, interpolation=cv2.INTER_LINEAR)
                img_gray_pred = cv2.cvtColor(img_resized_pred, cv2.COLOR_BGR2GRAY)
                img_normalized_pred = img_gray_pred.astype(np.float32) / 255.0
                frames_for_pred.append(img_normalized_pred)
            
            if frames_for_pred:
                # Convertit en tenseur : (T, C, H, W) où C=1 pour niveaux de gris
                sample_frames_tensor = torch.tensor(np.array(frames_for_pred), dtype=torch.float32).unsqueeze(1) 
                # Pas de vérité terrain pour les données externes, sauf si chargées séparément
                sample_text_tensor_truth = None 
                print(f"    [Prediction] Chargé {len(frames_for_pred)} images depuis {PATH_TO_YOUR_TEST_FRAMES_DIR}. Forme du tenseur : {sample_frames_tensor.shape}")
            else:
                print(f"ERREUR : Impossible de charger des images depuis {PATH_TO_YOUR_TEST_FRAMES_DIR}")
                sample_frames_tensor = None 
        
        # S'assure que la variable sample_frames_tensor est définie pour la suite de la cellule
        if sample_frames_tensor is None:
            print("Erreur lors du chargement des images, arrêt de la prédiction.")
            # Vous pouvez ajouter 'exit()' ici si vous exécutez comme un script
        # --- FIN DE LA MODIFICATION POUR LA CELLULE 8 ---
        ```
        L'utilisateur **doit** remplacer `"CHEMIN/VERS/VOTRE/DOSSIER/IMAGES_DE_TEST"` par le chemin réel vers son dossier d'images.

3.  **Exécuter la Cellule 8 Modifiée**:
    * Une fois les chemins correctement configurés et la partie chargement des données adaptée, exécutez la Cellule 8.
    * Elle chargera le modèle, prétraitera les images de votre dossier spécifié et affichera le texte prédit.

4.  **Expérimenter avec la Sortie (dans la Cellule 8)**:
    * Pour visualiser différents aspects de la prédiction, l'utilisateur peut modifier les paramètres de ``char_map_instance.indices_to_text(...)`` comme décrit dans les commentaires de la Cellule 8 (par ex., ``remove_blanks=False``, ``remove_duplicates=False``).
    * Pour afficher une image différente de sa séquence d'entrée (s'il a utilisé la modification ci-dessus), il peut changer l'index ``YOUR_FRAME_INDEX_HERE`` dans la section d'affichage de la Cellule 8 (par ex., ``frame_to_display = sample_frames_tensor[VOTRE_INDEX_D_IMAGE_ICI, 0, :, :].cpu().numpy()``).
