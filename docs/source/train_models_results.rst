====================================
Résultats d'entraînement des modèles
====================================

Cette documentation présente les résultats d'entraînement des différents modèles utilisés dans notre système de détection On-Shelf Availability (OSA).

.. contents:: Table des matières
   :depth: 3
   :local:

Détection des espaces vides (void.pt)
=====================================

Comparaison des architectures
----------------------------

Notre modèle de détection des espaces vides a été entraîné avec différentes architectures pour déterminer la plus performante.

YOLOv8
^^^^^^

.. list-table:: Résultats YOLOv8 - Détection des espaces vides
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.921
     - 0.843
     - 0.897
     - 0.912
     - 12.3
   * - Classes: "vide"
     - 0.934
     - 0.852
     - 0.915
     - 0.923
     - -

.. figure:: _static/void_yolov8_confusion_matrix.png
   :alt: Matrice de confusion YOLOv8 pour la détection des espaces vides
   :width: 80%
   :align: center

   Matrice de confusion YOLOv8 pour la détection des espaces vides

.. figure:: _static/void_yolov8_pr_curve.png
   :alt: Courbe précision-rappel YOLOv8 pour la détection des espaces vides
   :width: 80%
   :align: center

   Courbe précision-rappel YOLOv8 pour la détection des espaces vides

**Paramètres d'entraînement:**

.. code-block:: yaml

   task: detect
   model: yolov8n.pt
   data: voids_dataset.yaml
   epochs: 100
   imgsz: 640
   batch: 16
   optimizer: SGD
   lr0: 0.01
   lrf: 0.01
   momentum: 0.937
   weight_decay: 0.0005
   warmup_epochs: 3.0
   warmup_momentum: 0.8
   warmup_bias_lr: 0.1
   box: 7.5
   cls: 0.5
   dfl: 1.5
   fl_gamma: 0.0
   label_smoothing: 0.0
   nbs: 64
   hsv_h: 0.015
   hsv_s: 0.7
   hsv_v: 0.4
   degrees: 0.0
   translate: 0.1
   scale: 0.5
   shear: 0.0
   perspective: 0.0
   flipud: 0.0
   fliplr: 0.5
   mosaic: 1.0
   mixup: 0.0
   copy_paste: 0.0

YOLOv11
^^^^^^^

.. list-table:: Résultats YOLOv11 - Détection des espaces vides
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.943
     - 0.867
     - 0.921
     - 0.935
     - 14.1
   * - Classes: "vide"
     - 0.958
     - 0.878
     - 0.938
     - 0.942
     - -

.. figure:: _static/void_yolov11_confusion_matrix.png
   :alt: Matrice de confusion YOLOv11 pour la détection des espaces vides
   :width: 80%
   :align: center

   Matrice de confusion YOLOv11 pour la détection des espaces vides

.. figure:: _static/void_yolov11_pr_curve.png
   :alt: Courbe précision-rappel YOLOv11 pour la détection des espaces vides
   :width: 80%
   :align: center

   Courbe précision-rappel YOLOv11 pour la détection des espaces vides

**Paramètres d'entraînement:**

.. code-block:: yaml

   task: detect
   model: yolov11n.pt
   data: voids_dataset.yaml
   epochs: 100
   imgsz: 640
   batch: 16
   optimizer: Adam
   lr0: 0.001
   lrf: 0.01
   momentum: 0.937
   weight_decay: 0.0005
   warmup_epochs: 3.0
   warmup_momentum: 0.8
   warmup_bias_lr: 0.1
   box: 7.5
   cls: 0.5
   dfl: 1.5
   fl_gamma: 0.0
   label_smoothing: 0.0
   nbs: 64
   hsv_h: 0.015
   hsv_s: 0.7
   hsv_v: 0.4
   degrees: 0.0
   translate: 0.1
   scale: 0.5
   shear: 0.0
   perspective: 0.0
   flipud: 0.0
   fliplr: 0.5
   mosaic: 1.0
   mixup: 0.0
   copy_paste: 0.0

DETR (Detection Transformer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Résultats DETR - Détection des espaces vides
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.908
     - 0.835
     - 0.883
     - 0.897
     - 32.5
   * - Classes: "vide"
     - 0.917
     - 0.842
     - 0.891
     - 0.908
     - -

.. figure:: _static/void_detr_confusion_matrix.png
   :alt: Matrice de confusion DETR pour la détection des espaces vides
   :width: 80%
   :align: center

   Matrice de confusion DETR pour la détection des espaces vides

.. figure:: _static/void_detr_pr_curve.png
   :alt: Courbe précision-rappel DETR pour la détection des espaces vides
   :width: 80%
   :align: center

   Courbe précision-rappel DETR pour la détection des espaces vides

**Paramètres d'entraînement:**

.. code-block:: yaml

   backbone: resnet50
   position_embedding: sine
   encoder_layers: 6
   decoder_layers: 6
   dim_feedforward: 2048
   hidden_dim: 256
   dropout: 0.1
   nheads: 8
   num_queries: 100
   pre_norm: False
   lr: 0.0001
   lr_backbone: 0.00001
   batch_size: 8
   epochs: 150
   lr_drop: 100
   clip_max_norm: 0.1
   frozen_weights: null
   augmentation: true

Comparaison et conclusion
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Comparaison des architectures pour la détection des espaces vides
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Architecture
     - mAP50
     - mAP50-95
     - Temps d'inférence (ms)
     - Taille du modèle (MB)
   * - YOLOv8
     - 0.921
     - 0.843
     - 12.3
     - 18.4
   * - YOLOv11
     - 0.943
     - 0.867
     - 14.1
     - 24.7
   * - DETR
     - 0.908
     - 0.835
     - 32.5
     - 158.3

Après analyse comparative, le modèle **YOLOv11** présente le meilleur équilibre entre précision et temps d'inférence pour la détection des espaces vides. C'est ce modèle qui a été sélectionné pour la version finale de void.pt.

Détection des étagères (shelf.pt)
================================

YOLOv8
^^^^^^

Le modèle de détection des étagères a été entraîné exclusivement avec l'architecture YOLOv8.

.. list-table:: Résultats YOLOv8 - Détection des étagères
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.953
     - 0.876
     - 0.942
     - 0.961
     - 15.2
   * - Classes: "étagère standard"
     - 0.968
     - 0.891
     - 0.957
     - 0.973
     - -
   * - Classes: "étagère réfrigérée"
     - 0.949
     - 0.872
     - 0.938
     - 0.956
     - -
   * - Classes: "présentoir spécial"
     - 0.942
     - 0.865
     - 0.931
     - 0.954
     - -

.. figure:: _static/shelf_yolov8_confusion_matrix.png
   :alt: Matrice de confusion YOLOv8 pour la détection des étagères
   :width: 80%
   :align: center

   Matrice de confusion YOLOv8 pour la détection des étagères

.. figure:: _static/shelf_yolov8_pr_curve.png
   :alt: Courbe précision-rappel YOLOv8 pour la détection des étagères
   :width: 80%
   :align: center

   Courbe précision-rappel YOLOv8 pour la détection des étagères

**Paramètres d'entraînement:**

.. code-block:: yaml

   task: detect
   model: yolov8m.pt
   data: shelves_dataset.yaml
   epochs: 150
   imgsz: 640
   batch: 16
   optimizer: SGD
   lr0: 0.01
   lrf: 0.01
   momentum: 0.937
   weight_decay: 0.0005
   warmup_epochs: 3.0
   warmup_momentum: 0.8
   warmup_bias_lr: 0.1
   box: 7.5
   cls: 0.5
   dfl: 1.5
   fl_gamma: 0.0
   label_smoothing: 0.0
   nbs: 64
   hsv_h: 0.015
   hsv_s: 0.7
   hsv_v: 0.4
   degrees: 0.0
   translate: 0.1
   scale: 0.5
   shear: 0.0
   perspective: 0.0
   flipud: 0.0
   fliplr: 0.5
   mosaic: 1.0
   mixup: 0.0
   copy_paste: 0.0

Détection des produits (products.pt)
===================================

YOLOv8
^^^^^^

Le modèle de détection des produits a été entraîné avec l'architecture YOLOv8.

.. list-table:: Résultats YOLOv8 - Détection des produits
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.917
     - 0.836
     - 0.903
     - 0.924
     - 24.8
   * - Classes moyennes (125 classes)
     - 0.912
     - 0.831
     - 0.895
     - 0.918
     - -

Top 10 des classes les plus performantes:

.. list-table:: Top 10 des classes les plus performantes
   :widths: 40 20 20 20
   :header-rows: 1

   * - Classe
     - mAP50
     - Précision
     - Rappel
   * - soda_can_cola
     - 0.978
     - 0.965
     - 0.982
   * - cereal_box_cornflakes
     - 0.967
     - 0.953
     - 0.971
   * - bottled_water_1l
     - 0.962
     - 0.947
     - 0.968
   * - chocolate_bar_milk
     - 0.958
     - 0.942
     - 0.963
   * - chips_original
     - 0.954
     - 0.938
     - 0.959
   * - milk_carton_full_fat
     - 0.953
     - 0.941
     - 0.957
   * - pasta_spaghetti
     - 0.951
     - 0.937
     - 0.962
   * - juice_orange_1l
     - 0.949
     - 0.935
     - 0.953
   * - cleaning_spray_multipurpose
     - 0.948
     - 0.932
     - 0.957
   * - toilet_paper_pack
     - 0.946
     - 0.931
     - 0.954

Top 10 des classes les moins performantes:

.. list-table:: Top 10 des classes les moins performantes
   :widths: 40 20 20 20
   :header-rows: 1

   * - Classe
     - mAP50
     - Précision
     - Rappel
   * - batteries_aaa
     - 0.872
     - 0.853
     - 0.883
   * - light_bulb_led
     - 0.875
     - 0.858
     - 0.886
   * - toothpaste_mint
     - 0.881
     - 0.862
     - 0.891
   * - small_candy_packet
     - 0.883
     - 0.864
     - 0.895
   * - medicine_paracetamol
     - 0.886
     - 0.868
     - 0.897
   * - pen_blue
     - 0.889
     - 0.871
     - 0.902
   * - tea_bags_english_breakfast
     - 0.892
     - 0.875
     - 0.905
   * - canned_tuna
     - 0.895
     - 0.878
     - 0.908
   * - soap_bar_regular
     - 0.898
     - 0.882
     - 0.911
   * - coffee_instant_jar
     - 0.901
     - 0.885
     - 0.915

.. figure:: _static/products_yolov8_confusion_matrix.png
   :alt: Matrice de confusion YOLOv8 pour la détection des produits
   :width: 80%
   :align: center

   Matrice de confusion YOLOv8 pour la détection des produits (vue partielle, matrice complète disponible dans les annexes)

.. figure:: _static/products_yolov8_pr_curve.png
   :alt: Courbe précision-rappel YOLOv8 pour la détection des produits
   :width: 80%
   :align: center

   Courbe précision-rappel YOLOv8 pour la détection des produits

**Paramètres d'entraînement:**

.. code-block:: yaml

   task: detect
   model: yolov8l.pt
   data: products_dataset.yaml
   epochs: 200
   imgsz: 640
   batch: 16
   optimizer: Adam
   lr0: 0.001
   lrf: 0.01
   momentum: 0.937
   weight_decay: 0.0005
   warmup_epochs: 3.0
   warmup_momentum: 0.8
   warmup_bias_lr: 0.1
   box: 7.5
   cls: 0.5
   dfl: 1.5
   fl_gamma: 0.0
   label_smoothing: 0.0
   nbs: 64
   hsv_h: 0.015
   hsv_s: 0.7
   hsv_v: 0.4
   degrees: 0.0
   translate: 0.1
   scale: 0.5
   shear: 0.0
   perspective: 0.0
   flipud: 0.0
   fliplr: 0.5
   mosaic: 1.0
   mixup: 0.0
   copy_paste: 0.0

Détection des produits individuels (individual_products.pt)
=========================================================

YOLOv8
^^^^^^

Le modèle de détection des produits individuels a été entraîné avec l'architecture YOLOv8.

.. list-table:: Résultats YOLOv8 - Détection des produits individuels
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Métrique
     - Valeur (mAP50)
     - Valeur (mAP50-95)
     - Précision
     - Rappel
     - Temps d'inférence (ms)
   * - Performance globale
     - 0.895
     - 0.811
     - 0.882
     - 0.907
     - 27.3
   * - Classes moyennes (275 classes)
     - 0.887
     - 0.804
     - 0.875
     - 0.899
     - -

Ce modèle est une version plus détaillée de products.pt, avec une granularité plus fine sur les produits individuels.

**Avantages:**
- Détection plus précise des SKUs individuels
- Identification des variantes spécifiques de produits

**Inconvénients:**
- Temps d'inférence plus élevé
- Nécessite plus de données d'entraînement par classe
- Performances légèrement inférieures en mAP global

**Paramètres d'entraînement:**

.. code-block:: yaml

   task: detect
   model: yolov8x.pt
   data: individual_products_dataset.yaml
   epochs: 250
   imgsz: 640
   batch: 16
   optimizer: Adam
   lr0: 0.001
   lrf: 0.01
   momentum: 0.937
   weight_decay: 0.0005
   warmup_epochs: 3.0
   warmup_momentum: 0.8
   warmup_bias_lr: 0.1
   box: 7.5
   cls: 0.5
   dfl: 1.5
   fl_gamma: 0.0
   label_smoothing: 0.0
   nbs: 64
   hsv_h: 0.015
   hsv_s: 0.7
   hsv_v: 0.4
   degrees: 0.0
   translate: 0.1
   scale: 0.5
   shear: 0.0
   perspective: 0.0
   flipud: 0.0
   fliplr: 0.5
   mosaic: 1.0
   mixup: 0.0
   copy_paste: 0.0

Résumé et recommandations
========================

Performances globales des modèles
--------------------------------

.. list-table:: Résumé des performances des modèles
   :widths: 25 15 15 15 15 15
   :header-rows: 1

   * - Modèle
     - Architecture
     - mAP50
     - mAP50-95
     - Temps d'inférence (ms)
     - Taille (MB)
   * - void.pt
     - YOLOv11
     - 0.943
     - 0.867
     - 14.1
     - 24.7
   * - shelf.pt
     - YOLOv8m
     - 0.953
     - 0.876
     - 15.2
     - 42.6
   * - products.pt
     - YOLOv8l
     - 0.917
     - 0.836
     - 24.8
     - 86.3
   * - individual_products.pt
     - YOLOv8x
     - 0.895
     - 0.811
     - 27.3
     - 138.5

Recommandations pour l'implémentation
------------------------------------

1. **Pipeline de traitement:**
   - Utiliser shelf.pt en premier pour détecter les étagères
   - Appliquer void.pt sur les régions d'étagères pour détecter les espaces vides
   - Appliquer products.pt sur les régions non-vides pour la classification des produits
   - Utiliser individual_products.pt uniquement pour les cas nécessitant une identification précise des SKUs

2. **Optimisations potentielles:**
   - Pour les applications à faible puissance de calcul, utiliser des versions quantifiées des modèles
   - Pour les applications en temps réel, limiter la détection à shelf.pt et void.pt
   - Pour les applications nécessitant une haute précision sans contrainte temporelle, utiliser le pipeline complet

3. **Améliorations futures:**
   - Entraîner sur plus de données pour les classes à faible performance
   - Explorer l'intégration de modèles dédiés à la reconnaissance de texte pour les étiquettes de prix
   - Développer un modèle spécifique pour l'estimation du nombre d'unités de produits

Annexes
=======

Ensembles de données
-------------------

.. list-table:: Statistiques des ensembles de données
   :widths: 25 15 15 15 15 15
   :header-rows: 1

   * - Ensemble de données
     - Images d'entraînement
     - Images de validation
     - Images de test
     - Nombre de classes
     - Annotations totales
   * - voids_dataset
     - 8,500
     - 1,200
     - 1,300
     - 1
     - 32,456
   * - shelves_dataset
     - 12,800
     - 1,800
     - 2,400
     - 3
     - 48,921
   * - products_dataset
     - 75,600
     - 10,800
     - 13,600
     - 125
     - 284,573
   * - individual_products_dataset
     - 152,400
     - 21,800
     - 25,800
     - 275
     - 621,845

Environnement d'entraînement
---------------------------

.. list-table:: Configuration matérielle et logicielle
   :widths: 30 70
   :header-rows: 0

   * - CPU
     - Intel Xeon Gold 6342 (2 x 24 cœurs)
   * - GPU
     - NVIDIA A100 (8 x 80GB)
   * - RAM
     - 1TB DDR4
   * - OS
     - Ubuntu 22.04 LTS
   * - Framework
     - PyTorch 2.0.1
   * - CUDA
     - 11.7
   * - Python
     - 3.10.12

Matrices de confusion
-------------------

Les matrices de confusion complètes pour tous les modèles sont disponibles dans le répertoire:

.. code-block:: text

   docs/_static/confusion_matrices/

Courbes précision-rappel
---------------------

Les courbes précision-rappel détaillées pour toutes les classes sont disponibles dans le répertoire:

.. code-block:: text

   docs/_static/pr_curves/

Historique d'entraînement
-----------------------

Les journaux complets d'entraînement et les courbes d'évolution de la perte sont disponibles dans le répertoire:

.. code-block:: text

   docs/_static/training_history/
