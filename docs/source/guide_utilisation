Guide d’Utilisation du Système de Détection
===========================================

Ce guide explique comment utiliser ce projet via les notebooks fournis, en téléchargeant les modèles et fichiers nécessaires depuis Google Drive.

Aucune installation locale n’est requise : vous pouvez tout exécuter via **Google Colab**.

Prérequis
---------

- Navigateur web
- Connexion Internet
- Un compte Google pour utiliser Google Colab

Étapes d’Utilisation
--------------------

1. **Accéder au Dossier Google Drive**

   Cliquez sur le lien suivant pour accéder au dossier contenant les fichiers du projet :

   `https://drive.google.com/drive/folders/1Sfbmq9lNk3rqeiNyVtmlr4JQEQc-q2sN?usp=drive_link`

   Ce dossier contient :

   - `CV_project_marjan.ipynb` (notebook principal)
   - `shelf.pt`, `void.pt`, `products.pt`, `individual_products.pt` (modèles préentraînés)

2. **Ouvrir le Notebook dans Google Colab**

   - Une fois ouvert, vous pouvez exécuter les cellules une par une.

3. **Monter Google Drive dans le Notebook**

   Dès les premières cellules, vous serez invité à **monter votre Drive** pour accéder aux modèles :

   .. code-block:: python

       from google.colab import drive
       drive.mount('/content/drive')

   Ensuite, utilisez les chemins vers vos fichiers (par exemple) :

   .. code-block:: python

       MODEL_PATH = "/content/drive/MyDrive/mon_projet/models/shelf.pt"

4. **Lancer la Détection**

   Les cellules du notebook permettent de :

   - Charger les modèles (`.pt`)
   - Importer des images
   - Appliquer la détection (étagères, vides, produits)
   - Visualiser les résultats directement dans le notebook

Organisation Typique
--------------------

::

    Mon Drive/
    └── CV_project/
        ├── detection.ipynb
        ├── models/
        │   ├── shelf.pt
        │   ├── void.pt
        │   ├── products.pt
        │   └── individual_products.pt
        └── images/
            └── exemple.jpg

Résultats
---------

Les résultats sont affichés automatiquement dans le notebook. Vous pouvez aussi enregistrer les images annotées avec :

.. code-block:: python

    cv2.imwrite("output.jpg", annotated_image)

Support
-------

- Pour toute question, me contactez abderrahmanessafi133@gamil.com .


