# Partie 2 - Reconnaissance de séquence vidéo par marqueurs

## Propriétés attendues d'un marqueur

Un bon marqueur doit être discriminant (rare hors de la séquence recherchée), robuste aux changements raisonnables de compression, luminosité et échelle, rapide à comparer, et suffisamment stable sur plusieurs images. Un marqueur isolé est moins fiable ; plusieurs marqueurs répartis dans le temps réduisent les faux positifs.

## Approche retenue

La méthode utilise les descripteurs ORB. Hors ligne, `BuildMarkerDatabase.py` prélève régulièrement des images d'une séquence de référence et sauvegarde leurs descripteurs binaires ORB. En ligne, `RecognizeSequence.py` extrait ORB sur la vidéo cible et compare ces descripteurs avec une distance de Hamming. Le test de ratio de Lowe élimine les correspondances ambiguës. Une occurrence est retenue quand le nombre de bonnes correspondances dépasse un seuil, puis les détections proches sont regroupées.

ORB est choisi car il est gratuit, rapide, invariant à la rotation et relativement robuste aux changements d'échelle. Ses limites sont les images peu texturées, les transformations très importantes et les recadrages sévères.

## Architecture

```text
Séquence de référence -> extraction ORB -> base de marqueurs (descripteurs + manifest JSON)
Vidéo cible -> extraction ORB -> comparaison Hamming + ratio de Lowe -> occurrences JSON
```

La construction de la base est hors ligne. La recherche peut être effectuée sur un flux ou une vidéo cible, de manière périodique avec `--scan-step`.

## Exécution

Construire une base sur un extrait de 10 secondes (images 0 à 249 à 25 FPS) :

```bash
.venv/bin/python BuildMarkerDatabase.py Pub_C+_352_288_1.mp4 --name pub_cplus --start 0 --end 250 --step 25 --output marker_database
```

Les valeurs par défaut permettent aussi d'exécuter simplement `BuildMarkerDatabase.py` sans argument ; dans ce cas toute la vidéo est utilisée comme référence.

Rechercher cette séquence dans une vidéo :

```bash
.venv/bin/python RecognizeSequence.py marker_database Pub_C+_352_288_1.mp4
```

Après avoir créé `marker_database`, la commande `RecognizeSequence.py` sans argument utilise automatiquement cette base et la vidéo fournie.

Le fichier `sequence_detections.json` contient les images et instants détectés. Les paramètres `--scan-step`, `--min-good-matches` et `--ratio` doivent être ajustés selon les faux positifs et faux négatifs observés.
