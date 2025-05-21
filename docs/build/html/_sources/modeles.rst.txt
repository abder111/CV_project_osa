Modèles Préentraînés
====================

Le système utilise quatre modèles de détection préentraînés basés sur YOLOv8 ou une architecture similaire :

shelf.pt
--------

Modèle spécialisé dans la détection et la délimitation des étagères.

**Caractéristiques :**

- Détection de différents types d'étagères (standard, réfrigérées, présentoirs spéciaux).
- Gestion des occlusions partielles.
- Précision élevée même dans des conditions d'éclairage variables.

void.pt
-------

Modèle dédié à l'identification des espaces vides.

**Caractéristiques :**

- Détection précise des emplacements sans produit.
- Différenciation entre espace vide intentionnel et rupture de stock.
- Prise en compte des étiquettes de prix et des séparateurs.

products.pt
-----------

Modèle pour la reconnaissance et la classification des produits.

**Caractéristiques :**

- Identification de nombreuses catégories de produits.
- Gestion des variations d'emballage et d'orientation.
- Capacité à distinguer les différentes variantes d'un même produit.

individual_products.pt
-----------------------

Modèle spécialisé dans la détection des produits individuels (SKU - Stock Keeping Unit).

**Caractéristiques :**

- Identification fine des références précises de produits.
- Capacité à distinguer les variantes d’un même produit (taille, parfum, format).
- Utile pour les tâches de réassort automatique ou de vérification de planogramme.
- Haute précision pour des cas de classification détaillée dans des environnements complexes.

