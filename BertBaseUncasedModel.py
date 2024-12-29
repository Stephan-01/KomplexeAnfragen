import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


# Set a random seed
random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')

# Function to read sentence pairs from a text file
def read_sentence_pairs(file_path):
    sentence_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence1, sentence2 = line.strip().split('|')
            sentence_pairs.append((sentence1, sentence2))
    return sentence_pairs

# Read the sentence pairs from the file
sentence_pairs = read_sentence_pairs('4ov3.5NEU.txt')

# Initialize a list to store similarity scores
similarity_scores = []

# Iterate over each pair of sentences
for sentence1, sentence2 in sentence_pairs:
    # Tokenize and encode the sentences
    encodings = tokenizer.batch_encode_plus(
        [sentence1, sentence2],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Generate embeddings for the sentences
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

    # Compute cosine similarity between the sentence embeddings
    similarity_score = cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))

    # Append the similarity score to the list
    similarity_scores.append(similarity_score[0][0])

# Convert the similarity scores to a NumPy array for easy manipulation
similarity_scores = np.array(similarity_scores)

# Increase the number of bins for finer granularity
num_bins = 50

# Plot the distribution of cosine similarity scores with Matplotlib
plt.figure(figsize=(8, 6))
plt.hist(similarity_scores, bins=num_bins, range=(0, 1), color='blue', alpha=0.7, edgecolor='black')
plt.title('Verteilung der Kosinus-Ähnlichkeitswerte (BERT)')
plt.xlabel('Kosinus-Ähnlichkeitswert')
plt.ylabel('Häufigkeit')
plt.show()



print("Cosine Similarity Scores:", similarity_scores)