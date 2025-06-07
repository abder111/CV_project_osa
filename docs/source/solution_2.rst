Solution 2 : Analyse par Clustering et Feature Extraction
========================================================

Introduction
------------

Cette deuxième solution propose une approche innovante pour l'analyse de disponibilité des produits sur étagères (on-shelf availability) basée sur l'apprentissage non supervisé. Contrairement à l'approche multi-modèles de la Solution 1, cette méthode utilise un pipeline séquentiel qui combine détection YOLO, extraction de caractéristiques visuelles, réduction dimensionnelle et clustering automatique.

**Principe clé** : Détecter les produits individuels, extraire leurs caractéristiques visuelles, puis les regrouper automatiquement par similarité pour analyser la disponibilité et la répartition des catégories de produits sur les étagères.

**Avantage principal** : Catégorisation automatique des produits sans nécessiter d'étiquetage préalable, permettant une analyse flexible et adaptable à différents assortiments de produits.

Schéma de l'Architecture
------------------------

.. code-block:: text

    ┌─────────────────┐
    │   IMAGE INPUT   │
    │  (Shelf Photo)  │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────┐
    │              ÉTAPE 1: DÉTECTION YOLO               │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │         YOLO Object Detection                   ││
    │  │         (individual_products.pt)                ││
    │  │         Confidence: 0.5                         ││
    │  └─────────────────┬───────────────────────────────┘│
    └────────────────────┼─────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │              ÉTAPE 2: CROPPING                      │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │     Extract Bounding Boxes                      ││
    │  │     Crop Individual Products                    ││
    │  │     → /content/data/base/crops/object/          ││
    │  └─────────────────┬───────────────────────────────┘│
    └────────────────────┼─────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │         ÉTAPE 3: FEATURE EXTRACTION                 │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │      Deep Learning Feature Extraction           ││
    │  │                                                 ││
    │  │  ┌─────────────┐    ┌─────────────────────────┐ ││
    │  │  │ Img2Vec     │ OR │ ResNet18 PyTorch        │ ││
    │  │  │ (Primary)   │    │ (Fallback)              │ ││
    │  │  └─────────────┘    └─────────────────────────┘ ││
    │  │                                                 ││
    │  │  → High-dimensional feature vectors             ││
    │  └─────────────────┬───────────────────────────────┘│
    └────────────────────┼─────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │       ÉTAPE 4: DIMENSIONALITY REDUCTION             │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │              t-SNE Algorithm                    ││
    │  │                                                 ││
    │  │  • n_components = 3                             ││
    │  │  • perplexity = min(30, len(data)-1)            ││
    │  │  • Preserve local structure                     ││
    │  │                                                 ││
    │  │  → 3D embeddings for visualization              ││
    │  └─────────────────┬───────────────────────────────┘│
    └────────────────────┼─────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │           ÉTAPE 5: CLUSTERING                       │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │              K-Means Clustering                 ││
    │  │                                                 ││
    │  │  • Elbow method for optimal K                   ││
    │  │  • Automatic product categorization             ││
    │  │  • Cluster assignment for each product          ││
    │  └─────────────────┬───────────────────────────────┘│
    └────────────────────┼─────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │              RÉSULTATS FINAUX                       │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │  • Visualisation scatter plot 3D                ││
    │  │  • Clustering coloré par catégorie              ││
    │  │  • Images organisées par cluster                ││
    │  │  • Statistiques de distribution                 ││
    │  └─────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────┘

Pipeline de Traitement Détaillé
--------------------------------

Étape 1 : Détection YOLO et Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Identifier et localiser tous les produits individuels présents sur l'étagère.

.. code-block:: python

    # Configuration du modèle YOLO
    model = YOLO('individual_products.pt')
    results = model.predict(image_path, conf=0.5, save=False)
    
    # Extraction des boîtes englobantes
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()

**Caractéristiques techniques** :

* **Modèle** : YOLO pré-entraîné sur produits individuels
* **Seuil de confiance** : 0.5 pour équilibrer précision/rappel
* **Sortie** : Coordonnées de boîtes englobantes pour chaque produit détecté

Étape 2 : Cropping et Préparation des Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Isoler chaque produit dans une image séparée pour l'analyse individuelle.

.. code-block:: python

    # Cropping des produits détectés
    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        cropped_image = original_image[y1:y2, x1:x2]
        cv2.imwrite(f'/content/data/base/crops/object/product_{i}.jpg', cropped_image)

**Avantages** :

* **Focus** : Analyse concentrée sur le produit sans bruit de fond
* **Standardisation** : Images de taille cohérente pour l'extraction de caractéristiques
* **Organisation** : Stockage structuré pour les étapes suivantes

Étape 3 : Extraction de Caractéristiques Visuelles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Convertir les images de produits en vecteurs de caractéristiques numériques pour permettre l'analyse quantitative.

**Architecture Flexible** :

.. code-block:: python

    # Approche principale : Img2Vec
    try:
        from img2vec_pytorch import Img2Vec
        img2vec = Img2Vec(cuda=True)
        features = img2vec.get_vec(image)
    except:
        # Fallback : ResNet18 PyTorch
        model = models.resnet18(pretrained=True)
        features = extract_features_resnet(image, model)

**Spécifications techniques** :

* **Modèle principal** : Img2Vec (basé sur ResNet/VGG)
* **Modèle de secours** : ResNet18 PyTorch
* **Sortie** : Vecteurs de caractéristiques haute dimension (512-2048 dimensions)
* **Optimisation** : Utilisation GPU si disponible

Étape 4 : Réduction Dimensionnelle t-SNE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Projeter les vecteurs haute dimension dans un espace 3D pour la visualisation et le clustering.

.. code-block:: python

    # Configuration t-SNE
    tsne = TSNE(
        n_components=3,
        perplexity=min(30, len(embeddings)-1),
        random_state=42,
        n_iter=1000
    )
    
    # Transformation des données
    embeddings_3d = tsne.fit_transform(feature_vectors)

**Paramètres clés** :

* **n_components** : 3 (visualisation 3D optimale)
* **perplexity** : Adaptative selon la taille des données
* **conservation** : Structure locale des similarités préservée

Étape 5 : Clustering K-Means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objectif** : Regrouper automatiquement les produits similaires en catégories.

.. code-block:: python

    # Détermination du nombre optimal de clusters
    def find_optimal_clusters(embeddings, max_k=10):
        sse = []
        for k in range(1, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            sse.append(kmeans.inertia_)
        return optimal_k
    
    # Application du clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_3d)

**Méthodes d'optimisation** :

* **Méthode du coude** : Détermination automatique du nombre optimal de clusters
* **Stabilité** : Random state fixe pour la reproductibilité
* **Scalabilité** : Adapté aux datasets de taille variable

Visualisation et Analyse des Résultats
--------------------------------------

Graphique de Clustering 3D
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Génération** : Scatter plot 3D avec coloration par cluster

.. code-block:: python

    # Visualisation interactive
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Points colorés par cluster
    for cluster_id in range(n_clusters):
        cluster_points = embeddings_3d[cluster_labels == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  cluster_points[:, 2], label=f'Cluster {cluster_id}')

**Caractéristiques** :

* **Couleurs distinctes** : Un code couleur par catégorie de produits
* **Légende interactive** : Identification claire des clusters
* **Rotation 3D** : Exploration multi-angle des regroupements

Organisation des Images par Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Structure de sortie** :

.. code-block:: text

    /clustered_images/
    ├── cluster_0_[nom_personnalisé]/
    │   ├── product_001.jpg
    │   ├── product_015.jpg
    │   └── ...
    ├── cluster_1_[nom_personnalisé]/
    │   ├── product_003.jpg
    │   ├── product_022.jpg
    │   └── ...
    └── cluster_2_[nom_personnalisé]/
        ├── product_007.jpg
        └── ...

**Fonctionnalités** :

* **Nommage personnalisé** : Possibilité d'attribuer des noms métier aux clusters
* **Organisation automatique** : Tri des images selon leur appartenance
* **Statistiques** : Nombre de produits par catégorie

Métriques et Statistiques
-------------------------

Distribution des Clusters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyse de la distribution
    cluster_counts = Counter(cluster_labels)
    
    print("Distribution des produits par cluster:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(cluster_labels)) * 100
        print(f"Cluster {cluster_id}: {count} produits ({percentage:.1f}%)")

**Métriques disponibles** :

* **Taille des clusters** : Nombre de produits par catégorie
* **Répartition relative** : Pourcentages de distribution
* **Qualité du clustering** : Score de silhouette et inertie

Analyse de Disponibilité
~~~~~~~~~~~~~~~~~~~~~~~~

**Applications pratiques** :

1. **Détection de ruptures** : Clusters avec peu de produits
2. **Analyse d'assortiment** : Répartition des catégories
3. **Optimisation placement** : Regroupement des produits similaires
4. **Surveillance continue** : Évolution des distributions dans le temps

Avantages de cette Approche
---------------------------

✅ **Apprentissage non supervisé** : Pas besoin d'étiquetage manuel préalable

✅ **Flexibilité** : Adaptation automatique à nouveaux produits

✅ **Visualisation intuitive** : Compréhension immédiate des regroupements

✅ **Scalabilité** : Traitement efficace de grandes quantités de produits

✅ **Modularité** : Composants interchangeables selon les besoins

Limitations et Considérations
-----------------------------

⚠️ **Qualité de détection** : Dépendante de la performance du modèle YOLO

⚠️ **Paramétrage t-SNE** : Sensibilité aux hyperparamètres

⚠️ **Interprétation clusters** : Nécessite validation métier

⚠️ **Produits similaires** : Risque de confusion entre variantes proches

⚠️ **Stabilité** : Clustering peut varier selon les conditions

Applications Pratiques
----------------------

Cas d'Usage Retail
~~~~~~~~~~~~~~~~~~

1. **Audit automatique** : Vérification de l'assortiment présent
2. **Analyse concurrentielle** : Comparaison avec assortiment théorique
3. **Optimisation merchandising** : Regroupement optimal des produits
4. **Détection d'anomalies** : Identification de produits mal placés

Configuration Technique
-----------------------

Dépendances Requises
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ultralytics>=8.0.0
    torch>=1.9.0
    torchvision>=0.10.0
    scikit-learn>=1.0.0
    matplotlib>=3.5.0
    opencv-python>=4.5.0
    img2vec-pytorch>=1.0.0
    numpy>=1.21.0

Ressources Système
~~~~~~~~~~~~~~~~~~

* **GPU** : NVIDIA avec CUDA pour accélération (recommandé)
* **RAM** : 16 GB minimum pour datasets importants
* **Stockage** : Espace suffisant pour images croppées
* **CPU** : Processeur multi-core pour t-SNE

Paramètres Configurables
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    CONFIG = {
        'yolo_confidence': 0.5,
        'tsne_perplexity': 30,
        'tsne_components': 3,
        'kmeans_max_clusters': 10,
        'feature_extractor': 'img2vec',  # ou 'resnet18'
        'gpu_acceleration': True
    }

Intégration et Déploiement
--------------------------

Cette solution peut être intégrée dans différents environnements :

* **Pipeline batch** : Traitement périodique d'images d'étagères
* **API REST** : Service web pour analyse à la demande
* **Application mobile** : Analyse en temps réel sur le terrain
* **Système de monitoring** : Surveillance continue des rayons

Les résultats peuvent être exportés vers des formats standards (JSON, CSV) pour intégration avec des systèmes de gestion d'inventaire ou des tableaux de bord analytics.
