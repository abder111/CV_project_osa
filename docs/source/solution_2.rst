Solution 2 : Classification par Clustering Semi-Supervisé
========================================================

Introduction
------------

Cette solution propose une approche hybride innovante qui combine l'apprentissage non supervisé (clustering) avec l'apprentissage supervisé (CNN) pour créer un système de classification automatique des produits sur étagères. L'approche utilise le clustering comme méthode d'annotation semi-automatique, transformant ainsi un problème non supervisé en un pipeline d'apprentissage supervisé optimisé.

**Principe clé** : Utiliser le clustering intelligent pour générer automatiquement des annotations de qualité, puis entraîner un CNN spécialisé pour une classification rapide et précise des produits.

**Avantage principal** : Réduction drastique du coût d'annotation manuelle tout en maintenant une précision élevée de classification, permettant une mise à l'échelle rapide sur de nouveaux assortiments de produits.

Architecture de la Solution
---------------------------

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1: CLUSTERING                    │
│                      (Annotation Automatique)                  │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: Détection et Extraction des Produits                │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ IMAGE D'ENTRÉE  │ → │        YOLO DETECTION               │ │
│  │   (Étagère)     │    │   individual_products.pt            │ │
│  └─────────────────┘    │   Confidence: 0.5                   │ │
│                         └─────────────┬───────────────────────┘ │
│                                       │                         │
│                                       ▼                         │
│                         ┌─────────────────────────────────────┐ │
│                         │      CROPPING AUTOMATIQUE          │ │
│                         │   → /crops/product_000X.jpg        │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: Extraction de Caractéristiques et Clustering         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              FEATURE EXTRACTION                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────────────────────────────┐ │ │
│  │  │   Img2Vec   │ OR │         ResNet18 Features           │ │ │
│  │  │  (Primaire) │    │          (Fallback)                 │ │ │
│  │  └─────────────┘    └─────────────────────────────────────┘ │ │
│  │                                   │                         │ │
│  │                                   ▼                         │ │
│  │              ┌─────────────────────────────────────────┐    │ │
│  │              │         t-SNE REDUCTION                 │    │ │
│  │              │    • n_components = 3                   │    │ │
│  │              │    • Visualisation 3D                   │    │ │
│  │              └─────────────┬───────────────────────────┘    │ │
│  │                            │                                │ │
│  │                            ▼                                │ │
│  │              ┌─────────────────────────────────────────┐    │ │
│  │              │         K-MEANS CLUSTERING              │    │ │
│  │              │    • Méthode du coude                   │    │ │
│  │              │    • Clusters automatiques              │    │ │
│  │              └─────────────┬───────────────────────────┘    │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: Génération d'Annotations                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           ORGANISATION PAR CLUSTERS                         │ │
│  │                                                             │ │
│  │  /dataset/                                                  │ │
│  │  ├── cluster_0_boissons/                                    │ │
│  │  │   ├── product_001.jpg                                    │ │
│  │  │   ├── product_015.jpg                                    │ │
│  │  │   └── product_032.jpg                                    │ │
│  │  ├── cluster_1_snacks/                                      │ │
│  │  │   ├── product_003.jpg                                    │ │
│  │  │   └── product_021.jpg                                    │ │
│  │  └── cluster_2_produits_laitiers/                          │ │
│  │      ├── product_007.jpg                                    │ │
│  │      └── product_018.jpg                                    │ │
│  │                                                             │ │
│  │                           │                                 │ │
│  │                           ▼                                 │ │
│  │           ┌─────────────────────────────────────────┐       │ │
│  │           │    GÉNÉRATION ANNOTATIONS.JSON          │       │ │
│  │           │  • image_path → class_label             │       │ │
│  │           │  • Validation semi-automatique          │       │ │
│  │           └─────────────────────────────────────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2: APPRENTISSAGE                     │
│                    (Entraînement CNN)                          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: Préparation du Dataset d'Entraînement               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                DATASET STRUCTURE                            │ │
│  │                                                             │ │
│  │  /training_data/                                            │ │
│  │  ├── train/ (70%)                                           │ │
│  │  │   ├── boissons/                                          │ │
│  │  │   ├── snacks/                                            │ │
│  │  │   └── produits_laitiers/                                 │ │
│  │  ├── validation/ (20%)                                      │ │
│  │  │   ├── boissons/                                          │ │
│  │  │   ├── snacks/                                            │ │
│  │  │   └── produits_laitiers/                                 │ │
│  │  └── test/ (10%)                                            │ │
│  │      ├── boissons/                                          │ │
│  │      ├── snacks/                                            │ │
│  │      └── produits_laitiers/                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5: Entraînement CNN Optimisé                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ARCHITECTURE CNN LÉGÈRE                        │ │
│  │                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                  INPUT LAYER                            │ │ │
│  │  │                224x224x3 RGB                            │ │ │
│  │  └─────────────┬───────────────────────────────────────────┘ │ │
│  │                │                                             │ │
│  │                ▼                                             │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │           CONVOLUTIONAL BLOCKS                          │ │ │
│  │  │                                                         │ │ │
│  │  │  • Block 1: Conv2D(32) + BatchNorm + ReLU + MaxPool    │ │ │
│  │  │  • Block 2: Conv2D(64) + BatchNorm + ReLU + MaxPool    │ │ │
│  │  │  • Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPool   │ │ │
│  │  │  • Block 4: Conv2D(256) + BatchNorm + ReLU + MaxPool   │ │ │
│  │  └─────────────┬───────────────────────────────────────────┘ │ │
│  │                │                                             │ │
│  │                ▼                                             │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │           CLASSIFIER LAYERS                             │ │ │
│  │  │                                                         │ │ │
│  │  │  • GlobalAveragePooling2D                               │ │ │
│  │  │  • Dense(512) + Dropout(0.5)                           │ │ │
│  │  │  • Dense(256) + Dropout(0.3)                           │ │ │
│  │  │  • Dense(n_classes) + Softmax                          │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 3: DÉPLOIEMENT                       │
│                 (Classification en Production)                 │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE DE PRODUCTION                                         │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ NOUVELLE IMAGE  │ → │        YOLO DETECTION               │ │
│  │   (Étagère)     │    │         Cropping                    │ │
│  └─────────────────┘    └─────────────┬───────────────────────┘ │
│                                       │                         │
│                                       ▼                         │
│                         ┌─────────────────────────────────────┐ │
│                         │      CNN CLASSIFICATION             │ │
│                         │    • Prédiction en temps réel       │ │
│                         │    • Scores de confiance            │ │
│                         │    • Classification multi-classe    │ │
│                         └─────────────┬───────────────────────┘ │
│                                       │                         │
│                                       ▼                         │
│                         ┌─────────────────────────────────────┐ │
│                         │       RÉSULTATS FINAUX             │ │
│                         │  • Catégories identifiées          │ │
│                         │  • Localisation des produits       │ │
│                         │  • Analyse de disponibilité        │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Processus de Clustering Intelligent
-----------------------------------

Extraction de Caractéristiques Visuelles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L'extraction de caractéristiques constitue la base de notre approche de clustering. Elle transforme les images de produits en représentations numériques exploitables.

**Méthodes d'extraction** :

* **Img2Vec** (Méthode principale) : Utilise des réseaux pré-entraînés (ResNet, VGG) pour extraire des caractéristiques robustes
* **ResNet18** (Méthode de secours) : Alternative fiable en cas d'indisponibilité d'Img2Vec
* **Optimisation GPU** : Accélération matérielle pour traitement en lot

**Caractéristiques techniques** :

* Vecteurs de dimension 512-2048 selon le modèle utilisé
* Normalisation automatique des images d'entrée
* Extraction batch pour optimisation des performances

Réduction Dimensionnelle par t-SNE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La réduction dimensionnelle permet une visualisation efficace et améliore la qualité du clustering en éliminant le bruit dimensionnel.

**Configuration t-SNE** :

* **Composantes** : 3 dimensions pour visualisation 3D interactive
* **Perplexité** : Adaptative selon la taille du dataset (min 5, max 30)
* **Préservation** : Maintien des structures locales de similarité
* **Stabilité** : Random seed fixe pour reproductibilité

**Avantages** :

* Séparation claire des groupes de produits similaires
* Visualisation intuitive des relations inter-produits
* Réduction du bruit dans les données haute dimension

Clustering K-Means Optimisé
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le clustering automatique identifie les groupes naturels de produits sans supervision préalable.

**Détermination du nombre optimal de clusters** :

* **Méthode du coude** : Analyse de l'inertie pour différentes valeurs de K
* **Score de silhouette** : Validation de la cohésion intra-cluster
* **Contraintes métier** : Limitation selon l'assortiment attendu

**Paramètres de clustering** :

* Algorithme K-Means++ pour initialisation intelligente
* Maximum 15 clusters pour éviter la sur-segmentation
* Critères de convergence adaptatifs

Génération d'Annotations Semi-Automatiques
------------------------------------------

Organisation Hiérarchique des Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le système organise automatiquement les produits détectés selon leur appartenance aux clusters identifiés.

**Structure de données générée** :

```
dataset_clustered/
├── cluster_0_boissons_gazeuses/
│   ├── coca_cola_001.jpg
│   ├── pepsi_002.jpg
│   └── sprite_003.jpg
├── cluster_1_eaux_minerales/
│   ├── evian_004.jpg
│   ├── vittel_005.jpg
│   └── perrier_006.jpg
├── cluster_2_jus_fruits/
│   ├── tropicana_007.jpg
│   └── minute_maid_008.jpg
└── metadata/
    ├── cluster_analysis.json
    ├── confidence_scores.json
    └── visual_similarity.json
```

Validation et Raffinement
~~~~~~~~~~~~~~~~~~~~~~~~~

**Processus de validation** :

1. **Analyse de cohérence** : Vérification de la similarité visuelle intra-cluster
2. **Détection d'outliers** : Identification des produits mal classés
3. **Validation manuelle selective** : Contrôle sur échantillon représentatif
4. **Correction itérative** : Ajustement des clusters problématiques

**Métriques de qualité** :

* Score de silhouette moyen > 0.6
* Cohérence visuelle intra-cluster > 80%
* Taux de validation manuelle < 10%

Fichier d'Annotations Automatique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le système génère automatiquement un fichier d'annotations standardisé compatible avec les frameworks d'apprentissage supervisé.

**Format JSON généré** :

```json
{
  "dataset_info": {
    "total_images": 1250,
    "num_classes": 8,
    "creation_date": "2025-06-08",
    "clustering_method": "kmeans_tsne"
  },
  "class_mapping": {
    "0": "boissons_gazeuses",
    "1": "eaux_minerales", 
    "2": "jus_fruits",
    "3": "snacks_sales",
    "4": "chocolats",
    "5": "biscuits",
    "6": "produits_laitiers",
    "7": "conserves"
  },
  "annotations": [
    {
      "image_path": "dataset/cluster_0/coca_cola_001.jpg",
      "class_id": 0,
      "class_name": "boissons_gazeuses",
      "confidence_clustering": 0.89,
      "cluster_purity": 0.94
    }
  ]
}
```

Architecture CNN Optimisée
--------------------------

Conception du Modèle
~~~~~~~~~~~~~~~~~~~

Le CNN est spécialement conçu pour être léger et efficace tout en maintenant une précision élevée sur les catégories de produits identifiées par clustering.

**Principes de conception** :

* **Légèreté** : Nombre de paramètres optimisé pour déploiement mobile
* **Efficacité** : Architecture inspirée de MobileNet avec adaptations métier
* **Spécialisation** : Optimisation pour les caractéristiques des produits retail

**Couches convolutionnelles** :

* **Blocs convolutionnels** : 4 blocs avec augmentation progressive des filtres
* **Normalisation** : BatchNormalization après chaque convolution
* **Activation** : ReLU pour éviter le problème du gradient qui disparaît
* **Pooling** : MaxPooling2D pour réduction dimensionnelle contrôlée

**Tête de classification** :

* **Global Average Pooling** : Réduction drastique des paramètres
* **Couches denses** : 512 → 256 → n_classes avec dropout progressif
* **Activation finale** : Softmax pour probabilités de classe

Stratégie d'Entraînement
~~~~~~~~~~~~~~~~~~~~~~~~

**Préparation des données** :

* **Division** : 70% entraînement, 20% validation, 10% test
* **Augmentation** : Rotation, zoom, flip horizontal pour robustesse
* **Normalisation** : Standardisation selon ImageNet

**Hyperparamètres optimisés** :

* **Learning rate** : 0.001 avec décroissance adaptative
* **Batch size** : 32 pour équilibre mémoire/convergence
* **Epochs** : 50-100 avec early stopping
* **Optimiseur** : Adam avec beta1=0.9, beta2=0.999

**Techniques de régularisation** :

* **Dropout** : 0.5 première couche dense, 0.3 seconde couche
* **L2 regularization** : Coefficient 0.001 sur les couches denses
* **Early stopping** : Patience de 10 epochs sur validation loss

Métriques de Performance
~~~~~~~~~~~~~~~~~~~~~~~

**Évaluation du modèle** :

* **Précision globale** : Objectif > 95% sur test set
* **Précision par classe** : Équilibrage des performances inter-classes
* **Matrice de confusion** : Analyse détaillée des erreurs de classification
* **Temps d'inférence** : < 50ms par image sur GPU standard

Avantages de l'Approche Hybride
-------------------------------

Efficacité du Processus d'Annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Réduction des coûts** :

* **Annotation manuelle** : Seulement 5-10% du dataset nécessite validation
* **Temps de setup** : Division par 10 du temps de préparation
* **Scalabilité** : Addition facile de nouvelles catégories de produits

**Qualité des annotations** :

* **Cohérence** : Élimination des erreurs humaines d'étiquetage
* **Objectivité** : Critères de similarité quantifiés et reproductibles
* **Traçabilité** : Scores de confiance pour chaque annotation

Performance de Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Précision améliorée** :

* **Spécialisation** : CNN entraîné spécifiquement sur l'assortiment cible
* **Données équilibrées** : Clustering naturel évite les biais de classe
* **Features pertinentes** : Apprentissage focalisé sur caractéristiques discriminantes

**Vitesse d'exécution** :

* **Inférence rapide** : CNN léger optimisé pour temps réel
* **Batch processing** : Traitement parallèle de multiples produits
* **Optimisation matérielle** : Compatible GPU/CPU selon les ressources

Applications Pratiques
----------------------

Cas d'Usage Retail
~~~~~~~~~~~~~~~~~~

**Audit automatique d'assortiment** :

* Vérification de la présence des références attendues
* Détection des produits en rupture de stock
* Analyse de la conformité du planogramme

**Surveillance concurrentielle** :

* Identification des produits concurrents présents
* Analyse de la part de linéaire par marque
* Évolution temporelle de l'assortiment

**Optimisation merchandising** :

* Recommandations de placement basées sur les clusters
* Analyse des associations de produits
* Optimisation de la rotation des stocks

Intégration Système
~~~~~~~~~~~~~~~~~~

**API REST** :

* Endpoints pour upload d'images et récupération de résultats
* Format de réponse JSON standardisé
* Authentification et gestion des quotas

**Pipeline batch** :

* Traitement périodique d'images d'étagères
* Rapports automatisés de performance
* Historisation des données pour analyse de tendances

**Interface utilisateur** :

* Dashboard de visualisation des résultats
* Outils de validation et correction des annotations
* Export des données vers systèmes tiers (ERP, CRM)

Configuration et Déploiement
----------------------------

Environnement Technique
~~~~~~~~~~~~~~~~~~~~~~

**Dépendances système** :

* Python 3.8+ avec librairies ML standard
* PyTorch ou TensorFlow selon préférence
* OpenCV pour traitement d'images
* Scikit-learn pour clustering et métriques

**Ressources recommandées** :

* **GPU** : NVIDIA RTX 3060 ou supérieur pour entraînement
* **RAM** : 16GB minimum, 32GB recommandé
* **Stockage** : SSD 500GB pour datasets et modèles
* **CPU** : Processeur multi-core pour preprocessing

Paramètres Configurables
~~~~~~~~~~~~~~~~~~~~~~~~

**Configuration clustering** :

* Seuil de confiance YOLO : 0.3-0.7 selon qualité images
* Nombre max de clusters : 5-20 selon assortiment
* Perplexité t-SNE : 5-50 selon taille dataset

**Configuration CNN** :

* Architecture : Nombre de couches et filtres adaptables
* Augmentation de données : Intensité des transformations
* Hyperparamètres : Learning rate, batch size, regularization

Métriques de Suivi
~~~~~~~~~~~~~~~~~

**Phase clustering** :

* Score de silhouette des clusters
* Pureté intra-cluster (cohérence visuelle)
* Taux de validation manuelle nécessaire

**Phase entraînement** :

* Courbes de loss et accuracy
* Métriques par classe (precision, recall, F1-score)
* Temps de convergence et stability

**Phase production** :

* Latence d'inférence moyenne
* Précision en conditions réelles
* Taux de faux positifs/négatifs

Perspectives d'Évolution
-----------------------

Améliorations Techniques
~~~~~~~~~~~~~~~~~~~~~~~

**Auto-amélioration** :

* Feedback loop pour réentraînement périodique
* Active learning pour identifier les cas difficiles
* Adaptation continue aux nouveaux produits

**Optimisations performance** :

* Quantization des modèles pour déploiement edge
* Pruning des connexions non-critiques
* Techniques de distillation de connaissance

**Robustesse** :

* Augmentation de données adaptée au domaine retail
* Techniques d'adversarial training
* Gestion des conditions d'éclairage variables

Extensions Fonctionnelles
~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-modalité** :

* Intégration des informations textuelles (codes-barres, prix)
* Analyse des couleurs et formes géométriques
* Fusion avec données contextuelles (saison, promotion)

**Intelligence contextuelle** :

* Apprentissage des associations de produits
* Prédiction des ruptures de stock
* Recommandations de réassort intelligent

Cette solution hybride représente une avancée significative dans l'automatisation de l'annotation et de la classification des produits retail. Elle combine le meilleur des deux mondes : l'efficacité de l'apprentissage non supervisé pour l'annotation et la précision de l'apprentissage supervisé pour la classification en production.
