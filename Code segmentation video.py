import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# PARAMÈTRES
# ============================================================

VIDEO_PATH = r"C:\Users\Asus TUF\Downloads\BE1_Traitement_et_analyse_des_données_visuelles_et_sonores\Pub_C+_352_288_1.mp4"

# Seuil de détection des changements de plans
THRESHOLD = 0.35

# Nombre minimum de frames entre deux changements détectés
MIN_FRAME_DISTANCE = 5

# Largeur utilisée pour les calculs
# La vidéo originale n'est pas modifiée.
PROCESSING_WIDTH = 320

# Dossier où seront enregistrées les frames détectées
OUTPUT_FOLDER = "frames_detectees"


# ============================================================
# CRÉATION DU DOSSIER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 1. OUVERTURE DE LA VIDÉO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERREUR : impossible d'ouvrir la vidéo.")
    exit()


# ============================================================
# 2. INFORMATIONS SUR LA VIDÉO
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

nb_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

duration = nb_frames / fps if fps > 0 else 0


print("=" * 70)
print("INFORMATIONS SUR LA VIDÉO")
print("=" * 70)

print(f"Nombre total de frames : {nb_frames}")
print(f"Largeur                : {width} pixels")
print(f"Hauteur                : {height} pixels")
print(f"FPS                    : {fps:.2f}")
print(f"Durée                  : {duration:.2f} secondes")

print("=" * 70)


# ============================================================
# 3. LECTURE DE LA PREMIÈRE FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:
    print("ERREUR : impossible de lire la première frame.")
    cap.release()
    exit()


# ============================================================
# 4. AFFICHAGE DE LA PREMIÈRE FRAME
# ============================================================

first_frame_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 5))

plt.imshow(first_frame_rgb)

plt.title("Première frame de la vidéo")

plt.axis("off")
plt.close()

# ============================================================
# 5. FONCTION DE REDIMENSIONNEMENT
# ============================================================

def resize_frame(frame, new_width=PROCESSING_WIDTH):

    h, w = frame.shape[:2]

    new_height = int(
        h * new_width / w
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 6. HISTOGRAMME RGB
# ============================================================

def display_rgb_histogram(frame, frame_number):

    """
    Affiche les histogrammes Rouge, Vert et Bleu
    d'une frame.
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hist_r = cv2.calcHist(
        [rgb],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_g = cv2.calcHist(
        [rgb],
        [1],
        None,
        [256],
        [0, 256]
    )

    hist_b = cv2.calcHist(
        [rgb],
        [2],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hist_r,
        label="Rouge"
    )

    plt.plot(
        hist_g,
        label="Vert"
    )

    plt.plot(
        hist_b,
        label="Bleu"
    )

    plt.xlabel("Intensité des pixels")

    plt.ylabel("Nombre de pixels")

    plt.title(
        f"Histogramme RGB - Frame {frame_number}"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        f"histogramme_RGB_frame_{frame_number}.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

# ============================================================
# 7. AFFICHAGE DE L'HISTOGRAMME RGB
# ============================================================

display_rgb_histogram(
    first_frame,
    0
)


# ============================================================
# 8. HISTOGRAMME HSV
# ============================================================

def display_hsv_histogram(frame, frame_number):

    """
    Affiche les histogrammes H, S et V
    d'une frame.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hist_h = cv2.calcHist(
        [hsv],
        [0],
        None,
        [180],
        [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv],
        [1],
        None,
        [256],
        [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv],
        [2],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hist_h,
        label="Teinte (H)"
    )

    plt.plot(
        hist_s,
        label="Saturation (S)"
    )

    plt.plot(
        hist_v,
        label="Valeur (V)"
    )

    plt.xlabel("Valeur")

    plt.ylabel("Nombre de pixels")

    plt.title(
        f"Histogramme HSV - Frame {frame_number}"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        f"histogramme_HSV_frame_{frame_number}.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

# Affichage de l'histogramme HSV
display_hsv_histogram(
    first_frame,
    0
)


# ============================================================
# 9. MESURE BASÉE SUR LA COULEUR
# ============================================================

def color_measure(frame1, frame2):

    """
    Compare deux images à partir de leur information
    de couleur.

    Méthode :
        1. Conversion BGR -> HSV
        2. Calcul d'un histogramme H-S
        3. Normalisation
        4. Distance de Bhattacharyya

    Une valeur proche de 0 signifie que les images
    sont similaires.

    Une valeur plus élevée signifie qu'elles sont
    différentes.
    """

    hsv1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2HSV
    )

    hsv2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2HSV
    )

    # Histogramme H-S de la première image
    hist1 = cv2.calcHist(
        [hsv1],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    # Histogramme H-S de la deuxième image
    hist2 = cv2.calcHist(
        [hsv2],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    # Normalisation
    hist1 = cv2.normalize(
        hist1,
        hist1
    ).flatten()

    hist2 = cv2.normalize(
        hist2,
        hist2
    ).flatten()

    # Distance de Bhattacharyya
    distance = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_BHATTACHARYYA
    )

    return float(distance)


# ============================================================
# 10. MESURE BASÉE SUR LA GÉOMÉTRIE
# ============================================================

def geometry_measure(frame1, frame2):

    """
    Compare deux images à partir de leur information
    géométrique.

    Méthode :
        1. Conversion en niveaux de gris
        2. Détection des contours avec Canny
        3. Comparaison des cartes de contours
    """

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    # Détection des contours
    edges1 = cv2.Canny(
        gray1,
        100,
        200
    )

    edges2 = cv2.Canny(
        gray2,
        100,
        200
    )

    # Conversion entre 0 et 1
    edges1 = (
        edges1.astype(np.float32)
        / 255.0
    )

    edges2 = (
        edges2.astype(np.float32)
        / 255.0
    )

    # Différence absolue
    difference = np.abs(
        edges1 - edges2
    )

    # Moyenne des différences
    distance = np.mean(
        difference
    )

    return float(distance)


# ============================================================
# 11. EXTRACTION ET ANALYSE DE TOUTES LES FRAMES
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

frame_numbers = []

times = []

color_distances = []

geometry_distances = []

previous_frame = None

frame_index = 0


print("\n")
print("=" * 70)
print("ANALYSE DE TOUTES LES FRAMES")
print("=" * 70)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionnement pour les calculs
    processed_frame = resize_frame(
        frame
    )

    # Première frame
    if previous_frame is None:

        previous_frame = (
            processed_frame.copy()
        )

        frame_numbers.append(
            frame_index
        )

        times.append(
            frame_index / fps
        )

        color_distances.append(
            0
        )

        geometry_distances.append(
            0
        )

        frame_index += 1

        continue

    # --------------------------------------------------------
    # MESURE COULEUR
    # --------------------------------------------------------

    color_distance = color_measure(
        previous_frame,
        processed_frame
    )

    # --------------------------------------------------------
    # MESURE GÉOMÉTRIQUE
    # --------------------------------------------------------

    geometry_distance = geometry_measure(
        previous_frame,
        processed_frame
    )

    # --------------------------------------------------------
    # STOCKAGE
    # --------------------------------------------------------

    frame_numbers.append(
        frame_index
    )

    times.append(
        frame_index / fps
    )

    color_distances.append(
        color_distance
    )

    geometry_distances.append(
        geometry_distance
    )

    # La frame actuelle devient
    # la frame précédente
    previous_frame = (
        processed_frame.copy()
    )

    frame_index += 1


cap.release()


print(f"Nombre de frames analysées : {frame_index}")

print("=" * 70)


# ============================================================
# 12. NORMALISATION
# ============================================================

def normalize(values):

    values = np.array(
        values,
        dtype=float
    )

    minimum = np.min(values)

    maximum = np.max(values)

    if maximum - minimum == 0:

        return np.zeros_like(values)

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


color_normalized = normalize(
    color_distances
)

geometry_normalized = normalize(
    geometry_distances
)


# ============================================================
# 13. MESURE COMBINÉE
# ============================================================

combined_measure = (
    0.5 * color_normalized
    +
    0.5 * geometry_normalized
)


# ============================================================
# 14. DÉTECTION DES CHANGEMENTS DE PLANS
# ============================================================

shot_changes = []


for i in range(
    1,
    len(combined_measure)
):

    # Vérification du seuil
    if (
        combined_measure[i]
        > THRESHOLD
    ):

        # Première détection
        if len(shot_changes) == 0:

            shot_changes.append(i)

        else:

            # Évite plusieurs détections
            # pour le même changement
            if (
                i - shot_changes[-1]
                > MIN_FRAME_DISTANCE
            ):

                shot_changes.append(i)


# ============================================================
# 15. AFFICHAGE DES CHANGEMENTS DÉTECTÉS
# ============================================================

print("\n")
print("=" * 70)
print("CHANGEMENTS DE PLANS DÉTECTÉS")
print("=" * 70)


if len(shot_changes) == 0:

    print(
        "Aucun changement de plan détecté."
    )

else:

    for numero, index in enumerate(
        shot_changes,
        start=1
    ):

        time = index / fps

        value = (
            combined_measure[index]
        )

        print(
            f"{numero:03d} | "
            f"Frame : {index:5d} | "
            f"Temps : {time:7.2f} s | "
            f"Mesure : {value:.3f}"
        )


print("-" * 70)

print(
    "Nombre de changements détectés :",
    len(shot_changes)
)


# ============================================================
# 16. SAUVEGARDE DES FRAMES DÉTECTÉES
# ============================================================

print("\n")
print("=" * 70)
print("SAUVEGARDE DES FRAMES DÉTECTÉES")
print("=" * 70)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


for numero, change_index in enumerate(
    shot_changes,
    start=1
):

    # Aller directement à la frame
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        change_index
    )

    ret, frame = cap.read()

    if not ret:

        print(
            f"Impossible de lire "
            f"la frame {change_index}"
        )

        continue

    # Temps
    time = (
        change_index / fps
    )

    # Nom du fichier
    filename = (
        f"changement_{numero:03d}"
        f"_frame_{change_index:05d}"
        f"_{time:.2f}s.png"
    )

    # Chemin
    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    # Sauvegarde de la frame originale
    cv2.imwrite(
        filepath,
        frame
    )

    print(
        f"Frame {change_index} sauvegardée :"
    )

    print(
        f"    {filepath}"
    )


cap.release()


print("-" * 70)

print(
    f"{len(shot_changes)} frame(s) "
    "sauvegardée(s)"
)

print(
    f"Dossier : {OUTPUT_FOLDER}"
)


# ============================================================
# 17. COURBE DE LA MESURE COULEUR
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    times,
    color_normalized,
    label="Mesure couleur"
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    label=f"Seuil = {THRESHOLD}"
)


# Marquer les changements
for index in shot_changes:

    plt.axvline(
        times[index],
        linestyle=":",
        alpha=0.7
    )


plt.xlabel(
    "Temps (secondes)"
)

plt.ylabel(
    "Dissimilarité normalisée"
)

plt.title(
    "Dissimilarité basée sur la couleur"
)

plt.legend()

plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "courbe_couleur.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

# ============================================================
# 18. COURBE DE LA MESURE GÉOMÉTRIQUE
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    times,
    geometry_normalized,
    label="Mesure géométrique"
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    label=f"Seuil = {THRESHOLD}"
)


for index in shot_changes:

    plt.axvline(
        times[index],
        linestyle=":",
        alpha=0.7
    )


plt.xlabel(
    "Temps (secondes)"
)

plt.ylabel(
    "Dissimilarité normalisée"
)

plt.title(
    "Dissimilarité basée sur l'information géométrique"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "courbe_geometrique.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

# ============================================================
# 19. COURBE DE LA MESURE COMBINÉE
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    times,
    combined_measure,
    label="Mesure combinée"
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    label=f"Seuil = {THRESHOLD}"
)


# Affichage des changements
for index in shot_changes:

    plt.axvline(
        times[index],
        linestyle=":",
        alpha=0.7
    )

    plt.scatter(
        times[index],
        combined_measure[index]
    )


plt.xlabel(
    "Temps (secondes)"
)

plt.ylabel(
    "Dissimilarité"
)

plt.title(
    "Segmentation automatique de la vidéo"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "courbe_mesure_combinee.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

# ============================================================
# 20. CRÉATION DU TABLEAU DE RÉSULTATS
# ============================================================

results = pd.DataFrame({

    "Frame":
        frame_numbers,

    "Temps_s":
        times,

    "Distance_couleur":
        color_distances,

    "Distance_geometrique":
        geometry_distances,

    "Couleur_normalisee":
        color_normalized,

    "Geometrie_normalisee":
        geometry_normalized,

    "Mesure_combinee":
        combined_measure
})


# ============================================================
# 21. AJOUT DE LA COLONNE CHANGEMENT DE PLAN
# ============================================================

results["Changement_plan"] = False


for index in shot_changes:

    results.loc[
        index,
        "Changement_plan"
    ] = True


# ============================================================
# 22. SAUVEGARDE DU CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "resultats_segmentation.csv"
)

try:

    results.to_csv(
        csv_path,
        index=False
    )

    print("\nCSV sauvegardé avec succès :")
    print(csv_path)

except PermissionError:

    print("\nERREUR : le fichier CSV est déjà ouvert.")
    print(
        "Ferme 'resultats_segmentation.csv' "
        "puis relance le programme."
    )


print("\n")
print("=" * 70)
print("RÉSULTATS SAUVEGARDÉS")
print("=" * 70)

print(
    "Fichier CSV : "
    "resultats_segmentation.csv"
)

print(
    "Frames détectées : "
    f"{OUTPUT_FOLDER}/"
)


# ============================================================
# 23. ÉVALUATION AVEC LA VÉRITÉ TERRAIN
# ============================================================

TRUTH_FILE = os.path.join(
    os.path.dirname(VIDEO_PATH),
    "Vérité_terrain_Pub_C+_352_288_1.xlsx"
)


# ------------------------------------------------------------
# Lecture de la vérité terrain
# ------------------------------------------------------------

try:

    truth = pd.read_excel(
        TRUTH_FILE
    )

except FileNotFoundError:

    print("\nERREUR :")
    print(
        f"Le fichier '{TRUTH_FILE}' "
        "n'a pas été trouvé."
    )
    print(
        "Vérifie que le fichier Excel existe dans le dossier de la vidéo."
    )

    exit()


# ------------------------------------------------------------
# Extraction des frames de la vérité terrain
# ------------------------------------------------------------

# Première colonne = numéros des frames

ground_truth = pd.to_numeric(
    truth.iloc[:, 0],
    errors="coerce"
).dropna().astype(int).tolist()


# Suppression des doublons
ground_truth = sorted(
    list(set(ground_truth))
)


# ------------------------------------------------------------
# Extraction des frames détectées
# ------------------------------------------------------------

detected_frames = results.loc[
    results["Changement_plan"] == True,
    "Frame"
].astype(int).tolist()


detected_frames = sorted(
    list(set(detected_frames))
)


# ============================================================
# 24. TOLÉRANCE
# ============================================================

# Une détection située à +/- 5 frames
# de la vérité terrain est considérée
# comme une détection correcte.

TOLERANCE = 5


# ============================================================
# 25. COMPARAISON
# ============================================================

matched_ground_truth = set()

matches = []

false_positives = []


for detected in detected_frames:

    candidates = []


    for i, true_frame in enumerate(
        ground_truth
    ):

        # Cette vérité terrain a déjà
        # été associée à une détection
        if i in matched_ground_truth:

            continue


        difference = abs(
            detected - true_frame
        )


        if difference <= TOLERANCE:

            candidates.append(
                (
                    difference,
                    i,
                    true_frame
                )
            )


    # --------------------------------------------------------
    # Une correspondance a été trouvée
    # --------------------------------------------------------

    if candidates:

        # On choisit la vérité terrain
        # la plus proche

        candidates.sort(
            key=lambda x: x[0]
        )


        difference, index, true_frame = (
            candidates[0]
        )


        matched_ground_truth.add(
            index
        )


        matches.append({

            "Frame_verite_terrain":
                true_frame,

            "Frame_detectee":
                detected,

            "Erreur_frames":
                detected - true_frame
        })


    # --------------------------------------------------------
    # Aucune correspondance
    # --------------------------------------------------------

    else:

        false_positives.append(
            detected
        )


# ============================================================
# 26. CALCUL DES FAUX NÉGATIFS
# ============================================================

false_negatives = []


for i, true_frame in enumerate(
    ground_truth
):

    if i not in matched_ground_truth:

        false_negatives.append(
            true_frame
        )


# ============================================================
# 27. CALCUL DES PERFORMANCES
# ============================================================

VP = len(matches)

FP = len(false_positives)

FN = len(false_negatives)


# ------------------------------------------------------------
# Précision
# ------------------------------------------------------------

if VP + FP > 0:

    precision = (
        VP / (VP + FP)
    )

else:

    precision = 0


# ------------------------------------------------------------
# Rappel
# ------------------------------------------------------------

if VP + FN > 0:

    recall = (
        VP / (VP + FN)
    )

else:

    recall = 0


# ------------------------------------------------------------
# F1-score
# ------------------------------------------------------------

if precision + recall > 0:

    f1_score = (
        2
        * precision
        * recall
        /
        (precision + recall)
    )

else:

    f1_score = 0


# ============================================================
# 28. AFFICHAGE DES PERFORMANCES
# ============================================================

print("\n")
print("=" * 70)
print("ÉVALUATION AVEC LA VÉRITÉ TERRAIN")
print("=" * 70)


print(
    f"Vérité terrain : "
    f"{len(ground_truth)}"
)


print(
    f"Détections : "
    f"{len(detected_frames)}"
)


print(
    f"Vrais positifs (VP) : "
    f"{VP}"
)


print(
    f"Faux positifs (FP) : "
    f"{FP}"
)


print(
    f"Faux négatifs (FN) : "
    f"{FN}"
)


print(
    f"Précision : "
    f"{precision * 100:.2f}%"
)


print(
    f"Rappel : "
    f"{recall * 100:.2f}%"
)


print(
    f"F1-score : "
    f"{f1_score * 100:.2f}%"
)


print(
    f"Tolérance : "
    f"±{TOLERANCE} frames"
)


# ============================================================
# 29. SAUVEGARDE DU RÉSUMÉ DE L'ÉVALUATION
# ============================================================

evaluation_summary = pd.DataFrame({

    "Indicateur": [

        "Vérité terrain",

        "Détections",

        "Vrais positifs",

        "Faux positifs",

        "Faux négatifs",

        "Précision",

        "Rappel",

        "F1-score",

        "Tolérance"
    ],


    "Valeur": [

        len(ground_truth),

        len(detected_frames),

        VP,

        FP,

        FN,

        precision,

        recall,

        f1_score,

        TOLERANCE
    ]
})


evaluation_summary.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "evaluation_resume.csv"
    ),

    index=False
)


# ============================================================
# 30. SAUVEGARDE DES CORRESPONDANCES
# ============================================================

matches_df = pd.DataFrame(
    matches
)


matches_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "correspondances.csv"
    ),

    index=False
)


# ============================================================
# 31. SAUVEGARDE DES FAUX POSITIFS
# ============================================================

pd.DataFrame({

    "Frame":
        false_positives

}).to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "faux_positifs.csv"
    ),

    index=False
)


# ============================================================
# 32. SAUVEGARDE DES FAUX NÉGATIFS
# ============================================================

pd.DataFrame({

    "Frame":
        false_negatives

}).to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "faux_negatifs.csv"
    ),

    index=False
)


# ============================================================
# 33. RÉSUMÉ FINAL
# ============================================================

print("\n")
print("=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)


print(
    f"Nombre total de frames : "
    f"{nb_frames}"
)


print(
    f"Nombre de frames analysées : "
    f"{frame_index}"
)


print(
    f"FPS : "
    f"{fps:.2f}"
)


print(
    f"Durée : "
    f"{duration:.2f} secondes"
)


print(
    f"Seuil : "
    f"{THRESHOLD}"
)


print(
    f"Changements détectés : "
    f"{len(shot_changes)}"
)


print(
    f"Vérité terrain : "
    f"{len(ground_truth)}"
)


print("\n--- PERFORMANCES ---")


print(
    f"VP : "
    f"{VP}"
)


print(
    f"FP : "
    f"{FP}"
)


print(
    f"FN : "
    f"{FN}"
)


print(
    f"Précision : "
    f"{precision * 100:.2f}%"
)


print(
    f"Rappel : "
    f"{recall * 100:.2f}%"
)


print(
    f"F1-score : "
    f"{f1_score * 100:.2f}%"
)


print(
    f"Tolérance : "
    f"±{TOLERANCE} frames"
)


print("\n--- FICHIERS ---")


print(
    f"Frames détectées : "
    f"{OUTPUT_FOLDER}/"
)


print(
    "CSV principal : "
    "resultats_segmentation.csv"
)


print(
    "Évaluation : "
    "evaluation_resume.csv"
)


print(
    "Correspondances : "
    "correspondances.csv"
)


print(
    "Faux positifs : "
    "faux_positifs.csv"
)


print(
    "Faux négatifs : "
    "faux_negatifs.csv"
)


print("=" * 70)

print("\nAffichage des graphiques...")
plt.show()
