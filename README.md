Introduction
=================================================


Le système de détection On-Shelf Availability (OSA) est une solution avancée de vision par ordinateur conçue pour identifier automatiquement les espaces vides sur les rayons, détecter les étagères et reconnaître les produits présents dans un environnement commercial.

Dans le secteur de la grande distribution et du commerce de détail, les ruptures de stock représentent un enjeu majeur pouvant entraîner jusqu'à 4% de pertes de ventes annuelles. Notre solution technologique permet aux détaillants d'optimiser la gestion des stocks, d'améliorer l'expérience client et de maximiser les ventes en assurant une disponibilité optimale des produits.

Le système OSA utilise des algorithmes d'intelligence artificielle de pointe pour analyser en temps réel l'état des rayons et fournir des informations précieuses aux gestionnaires de magasins.

Objectifs du Système
--------------------

Le système a été développé avec des objectifs précis pour répondre aux besoins spécifiques du secteur de la distribution :

- **Détecter les espaces vides sur les étagères** : Identification automatique des zones nécessitant un réapprovisionnement
- **Identifier et localiser les étagères** : Cartographie précise de l'espace commercial pour un suivi optimal
- **Reconnaître les produits présents** : Classification des articles présents sur les rayons
- **Détecter les produits individuels** : Comptage précis des unités pour une gestion fine des stocks
- **Générer des rapports en temps réel** : Documentation détaillée sur l'état des stocks pour faciliter la prise de décision
- **Alerter le personnel** : Notifications automatiques en cas de rupture de stock pour une action rapide

Architecture du Système
-----------------------

Le système OSA repose sur une architecture modulaire comprenant quatre modèles de détection principaux qui fonctionnent en séquence pour assurer une analyse complète et précise des rayonnages.

.. code-block::

    ┌─────────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌───────────────────────────┐
    │                 │     │                │     │                 │     │                           │
    │  Détection des  │────▶│ Détection des  │────▶│ Détection des   │────▶│ Détection des produits    │
    │    étagères     │     │     vides      │     │    produits     │     │     individuels           │
    │   (shelf.pt)    │     │   (void.pt)    │     │  (products.pt)  │     │ (individual_products.pt)  │
    │                 │     │                │     │                 │     │                           │
    └─────────────────┘     └────────────────┘     └─────────────────┘     └───────────────────────────┘
             │                     │                       │                           │
             ▼                     ▼                       ▼                           ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                   Traitement des données                                      │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                 Génération des rapports                                       │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘

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

   Accéder au dossier Google Drive https://drive.google.com/drive/folders/1Sfbmq9lNk3rqeiNyVtmlr4JQEQc-q2sN?usp=drive_link

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
        ├── Cv_project_marjan.ipynb
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


