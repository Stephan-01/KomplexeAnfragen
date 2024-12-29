import matplotlib.pyplot as plt
import numpy as np

# Cosine Similarity Scores für zwei Modelle falls z.B: vergleich mit bertbase der nicht direkt möglich ist im anderen Script
similarity_scores_model1 = [1., 0.97623634, 0.96250826, 0.9770225, 0.9574619, 0.94557667, 0.93554604, 0.9674314]
similarity_scores_model2 = [1., 0.92895645, 0.8209047, 0.9400171, 0.74436337, 0.7335876, 0.85567856, 0.8664712 ]

# Konvertiere die Similarity Scores in NumPy-Arrays (falls benötigt)
similarity_scores_model1 = np.array(similarity_scores_model1)
similarity_scores_model2 = np.array(similarity_scores_model2)

# Anzahl der Bins für eine feinere Granularität
num_bins = 50

# Erstelle das Histogramm
plt.figure(figsize=(14, 6))

# Histogramm für das erste Modell
plt.subplot(1, 2, 1)
plt.hist(similarity_scores_model1, bins=num_bins, range=(0, 1), color='blue', alpha=0.7, edgecolor='black')
plt.title('Verteilung der Kosinus-Ähnlichkeitswerte (bert-base-uncased)')
plt.xlabel('Kosinus-Ähnlichkeitswert')
plt.ylabel('Häufigkeit')

# Histogramm für das zweite Modell
plt.subplot(1, 2, 2)
plt.hist(similarity_scores_model2, bins=num_bins, range=(0, 1), color='green', alpha=0.7, edgecolor='black')
plt.title('Verteilung der Kosinus-Ähnlichkeitswerte (all-MiniLM-L6-v2 )')
plt.xlabel('Kosinus-Ähnlichkeitswert')
plt.ylabel('Häufigkeit')

# Layout-Anpassung und Anzeige
plt.tight_layout()
plt.show()
