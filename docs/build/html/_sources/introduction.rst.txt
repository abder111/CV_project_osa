Introduction
============

Le projet **On-Shelf Availability** (OSA) vise à développer une solution intelligente de surveillance des rayons de supermarchés à l’aide de la vision par ordinateur.

L'objectif est de détecter automatiquement la disponibilité des produits en rayon et de signaler les vides (zones où un produit devrait être présent mais est manquant). Cette solution permet d'améliorer la gestion des stocks, la satisfaction client et l'efficacité des réapprovisionnements.

Nous utilisons des modèles de détection d’objets (YOLO, Faster R-CNN, etc.) pour identifier :
- les emplacements vides dans les rayons (zones sans produit),
- et les produits spécifiques (par exemple, 10 références connues).

Les images sont capturées à intervalles réguliers, analysées par notre modèle d’intelligence artificielle, puis les résultats sont présentés sous forme de visualisations (bounding boxes, pourcentage de vide, etc.).

Ce projet s'inscrit dans le cadre d'une initiative d'optimisation de la chaîne d'approvisionnement, en s’appuyant sur des technologies modernes comme :
- Python et OpenCV pour le traitement d’image,
- des frameworks d'apprentissage profond (YOLOv5, PyTorch, TensorFlow),
- et Read the Docs pour la documentation technique.

