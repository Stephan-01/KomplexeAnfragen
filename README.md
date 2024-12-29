# README

Dieses Repository enthält verschiedene Programme, die darauf abzielen, die Reproduzierbarkeit und Erweiterung der Ergebnisse der entsprechenden Arbeit „Optimierung der Antwortqualität von RAG-Systemantworten: Validierung und Komplexitätsbewältigung“ zu unterstützen. Die Programme bieten sowohl eine visuelle Darstellung als auch die Ausgabe der Ergebnisse in der Konsole.

## Übersicht der Programme

### 1. **BertBaseUncasedModel**
Dieses Programm dient dazu, das Modell `google-bert/bert-base-uncased` ([Hugging Face Link](https://huggingface.co/google-bert/bert-base-uncased)) direkt zu laden. Im Gegensatz zu anderen verwendeten Embedding-Modellen kann dieses Modell nicht über `SentenceTransformer` importiert werden.

### 2. **KosinusHistogram**
Mit diesem Programm können Embeddings eines spezifischen Modells mithilfe der Kosinusähnlichkeit analysiert werden. 
- Das Modell wird über `SentenceTransformer` importiert.
- Die berechneten Kosinuswerte werden in einem Histogramm visualisiert.

### 3. **VergleichEmbeddingModelle**
Dieses Programm ermöglicht den direkten Vergleich zweier Embedding-Modelle, die über `SentenceTransformer` importiert werden. Die Ergebnisse werden in der Konsole sowie in einer geeigneten Visualisierung dargestellt.

### 4. **VergleichNeuAlt**
Dieses Programm vergleicht alte Ergebnisse mit neuen Ergebnissen.
- **Erste Liste**: Enthält die Kosinuswerte, die z. B. durch das Programm „KosinusHistogram“ und das Modell `sentence-transformers/all-MiniLM-L6-v2` erzeugt wurden.
- **Zweite Liste**: Beinhaltet die subjektiven Bewertungen der „Ursprungsbewertungen“.

### 5. **VergleichEmbeddingModelle2**
Im Gegensatz zu „VergleichEmbeddingModelle“ werden hier die Kosinus-Ergebnisse verschiedener Modelle in separaten Listen gespeichert. Dies ermöglicht den Vergleich von Modellen, die nicht direkt importiert werden können (z. B. `bert-base-uncased`).

## Anpassbarkeit und Erweiterbarkeit
Die Programme sind modular und leicht erweiterbar. Beispielsweise kann die Anzahl der Vergleiche oder das genutzte Datenset schnell und einfach angepasst werden.

## Datenset
Alle Programme, die die Kosinusähnlichkeit mithilfe von Embeddings berechnen, verwenden das enthaltene Beispieldatenset `4ov3.5NEU.txt`. Dieses kann ebenfalls problemlos ausgetauscht werden. Wichtig ist, dass das Datenset im folgenden Format vorliegt:
Satz1|Satz2 - Die Trennung der Sätze durch das Zeichen `|` ist entscheidend.
