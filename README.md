# BE1 - Partie 0 : préparation avant la segmentation

Utiliser l'environnement déjà présent dans le dossier :

```bash
.venv/bin/python VideoReader.py
.venv/bin/python OpenVideo.py
.venv/bin/python ReadAllFrames.py
.venv/bin/python ShowFrames.py --index 100
```

Chaque script correspond à une tâche du sujet :

| Script | Tâche |
| --- | --- |
| `OpenVideo.py` | Ouvrir et lire la vidéo. Touche `q` pour quitter. |
| `VideoReader.py` | Obtenir les métadonnées : nombre d'images, taille, FPS et durée. |
| `ReadAllFrames.py` | Récupérer toutes les images en matrice NumPy `(N, H, W, 3)` BGR. |
| `ShowFrames.py` | Récupérer et afficher une seule image précise, avec sa matrice. |

La vidéo fournie contient 3 289 images de 352 x 288 pixels, à 25 images/s. Charger la matrice complète nécessite environ 1 Go de mémoire vive. Pour la sauvegarder :

```bash
.venv/bin/python ReadAllFrames.py
.venv/bin/python ReadAllFrames.py --print-frame-matrix 100
```

## Partie edges uniquement

Ces scripts ne réalisent aucune segmentation : ils produisent seulement des images de gradient et de contours à partir d'une image de la vidéo.

```bash
.venv/bin/python SobelGradient.py --index 100
.venv/bin/python CannyEdges.py --index 100
```

Les images PNG sont enregistrées dans `results/edges/`, avec fond blanc et contours noirs. Ajouter `--show` pour les afficher dans des fenêtres OpenCV.

## Partie 2 - Reconnaissance de séquence vidéo

La partie reconnaissance par marqueurs ORB est indépendante de la segmentation.
