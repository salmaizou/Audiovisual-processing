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

## Partie 2 - Segmentation vidéo

Méthode de détection des changements de plan
Objectif et difficultés
L’objectif est de détecter automatiquement les changements de plan en comparant les images successives d’une vidéo. Un changement de plan peut provoquer une variation importante de l’information visuelle. Cependant, une variation entre deux images ne signifie pas toujours qu’un plan a changé : un mouvement de caméra, un objet qui se déplace, un changement d’éclairage ou un effet visuel peuvent également modifier l’image.

À l’inverse, deux plans différents peuvent présenter des couleurs ou des formes similaires. Le choix des mesures de comparaison, du seuil de détection et de la tolérance utilisée pour l’évaluation influence donc les résultats.
Mesure fondée sur la couleur
Pour comparer les couleurs de deux images, chaque frame est convertie en espace HSV. Le programme calcule ensuite un histogramme bidimensionnel sur la teinte et la saturation, puis compare les histogrammes avec la distance de Bhattacharyya.

Cette distance mesure la différence entre les distributions de couleurs des deux frames : une valeur faible indique des distributions proches, tandis qu’une valeur élevée indique une différence plus importante.

Avantage : cette mesure résume efficacement la composition générale en couleurs et peut détecter des transitions accompagnées d’un changement visuel marqué.
Inconvénient : l’histogramme ne conserve pas la position des couleurs dans l’image. Deux images dont les éléments sont disposés différemment peuvent donc présenter des histogrammes similaires.

Mesure fondée sur l’information géométrique
Pour mesurer les différences géométriques, les images sont converties en niveaux de gris, puis les contours sont détectés avec l’algorithme de Canny. Le programme calcule ensuite la différence moyenne entre les deux cartes de contours.

Une différence faible signifie que les contours sont globalement similaires. Une différence élevée peut signaler une modification importante de la structure visuelle.
Avantage : cette mesure tient compte de la structure et des contours, même lorsque les couleurs sont proches.

Inconvénient : elle peut être sensible au bruit, aux mouvements d’objets, aux changements d’éclairage et aux paramètres de détection des contours. Ces variations peuvent provoquer des alertes sans véritable changement de plan.

Combinaison des mesures et détection
Les deux mesures sont normalisées entre 0 et 1, puis combinées avec un poids identique :
Combinaison des mesures
\[
\boxed{
D_{\text{combinée}}
= 0{,}5\,D_{\text{couleur normalisée}}
+ 0{,}5\,D_{\text{géométrie normalisée}}
}
\]
Le programme considère qu’un changement potentiel a lieu lorsque cette mesure combinée dépasse 0,35. Un écart minimal de 5 frames entre deux détections est également imposé afin de limiter les alertes répétées autour d’une même transition.

Ces valeurs sont des paramètres de l’expérience : elles peuvent être ajustées et comparées pour observer leur effet sur les résultats.
Critères d’évaluation
Les changements détectés par le programme sont comparés aux changements indiqués dans le fichier de vérité terrain. Une détection est considérée correcte si elle se trouve à ±5 frames d’un changement de référence.

L’évaluation distingue :

Vrais positifs (VP) : changements détectés correspondant à la vérité terrain ;
Faux positifs (FP) : changements détectés sans changement de référence correspondant ;
Faux négatifs (FN) : changements de référence non détectés.
La précision indique la proportion de détections qui sont correctes :

[
\text{Précision} = \frac{VP}{VP + FP}
]

Le rappel indique la proportion des changements de référence retrouvés :

[
\text{Rappel} = \frac{VP}{VP + FN}
]

Le F1-score résume la précision et le rappel par leur moyenne harmonique :

[
F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}
{\text{Précision} + \text{Rappel}}
]
La précision est utile lorsque l’on souhaite limiter les fausses détections. Le rappel est important lorsque l’on veut retrouver le plus grand nombre possible de changements réels. Le F1-score permet d’avoir une mesure synthétique tenant compte des deux.

Avantages et limites de l’approche
La combinaison de l’information de couleur et de l’information géométrique permet de prendre en compte deux aspects différents des frames. Le programme fournit également une courbe de mesure, les changements détectés et des métriques d’évaluation par rapport à une vérité terrain.

Cependant, les poids des deux mesures, le seuil de 0,35, la largeur de traitement de 320 pixels et la distance minimale de 5 frames influencent les résultats. Ils doivent être considérés comme des choix expérimentaux, et non comme des valeurs universelles. La méthode compare ici les frames successives : elle peut donc être perturbée par des mouvements ou des variations visuelles rapides.

Conclusion
Le programme constitue une première méthode de segmentation temporelle fondée sur la comparaison des frames successives. Il combine une mesure de couleur et une mesure géométrique, puis utilise un seuil pour proposer les changements de plan. Ses performances sont évaluées à l’aide d’une vérité terrain, avec une tolérance de ±5 frames.

Pour conclure sur son efficacité, il faut présenter les valeurs réellement obtenues de précision, de rappel et de F1-score. Il est également utile de comparer plusieurs seuils ou poids et d’examiner les faux positifs et les faux négatifs afin de déterminer dans quelles situations la méthode fonctionne bien ou échoue.
