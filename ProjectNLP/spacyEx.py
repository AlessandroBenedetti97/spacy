from collections import Counter
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.lang.it.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

# Carica il dataset utilizzando datasets.load_dataset
Cdataset = load_dataset("itacasehold/itacasehold", split='train')

# Estrai documenti e etichette dal dataset di addestramento
documents = Cdataset["summary"]
labels = Cdataset["materia"]

# Conta le occorrenze di ciascuna etichetta nel dataset
label_counts = Counter(labels)

# Trova etichette che compaiono almeno 5 volte
common_labels = [label for label, count in label_counts.items() if count >= 13]

# Filtra i documenti e le etichette mantenendo solo quelli con etichette comuni
filtered_indices = [i for i, label in enumerate(labels) if label in common_labels]
documents_filtered = [documents[i] for i in filtered_indices]
labels_filtered = [labels[i] for i in filtered_indices]

# Visualizza un istogramma delle frequenze dopo il filtraggio
label_counts_filtered = Counter(labels_filtered)
plt.bar(label_counts_filtered.keys(), label_counts_filtered.values())
plt.xlabel('Etichetta')
plt.ylabel('Frequenza')
plt.title('Frequenza delle etichette nel dataset (dopo il filtraggio)')
plt.xticks(rotation=45, ha="right")
plt.show()

# Rimuovi la punteggiatura, lemmatizza e rimuovi le stop words dal dataset di Spacy
nlp = spacy.load('it_core_news_lg')

documents_preprocessed = [
    " ".join([token.lemma_.lower() for token in nlp(doc) if not token.is_punct and token.text.lower() not in STOP_WORDS])
    for doc in documents_filtered
]

# Usa il TF-IDF per ottenere le rappresentazioni vettoriali
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents_preprocessed)
tfidf_embeddings = tfidf_matrix.toarray()

# Converte l'array numpy TF-IDF in un tensore PyTorch
embeddings_tensor = torch.tensor(tfidf_embeddings, dtype=torch.float32)

# Tratta labels come una lista di stringhe
labels_list = labels_filtered
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_list)
labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

# Suddividi il dataset di addestramento in set di addestramento, validazione e test
train_indices, test_indices = train_test_split(range(len(labels_encoded)), test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=42)

# Seleziona gli embeddings e le etichette di addestramento, validazione e test
train_embeddings = embeddings_tensor[train_indices]
train_labels = labels_tensor[train_indices]
val_embeddings = embeddings_tensor[val_indices]
val_labels = labels_tensor[val_indices]
test_embeddings = embeddings_tensor[test_indices]
test_labels = labels_tensor[test_indices]

# Definisci il tuo modello con due strati nascosti
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Parametri del modello
input_size = tfidf_embeddings.shape[1]  # Dimensione con TF-IDF
hidden_size = 64
output_size = len(set(labels_filtered))

# Inizializza il modello
model = SimpleClassifier(input_size, hidden_size, output_size, dropout_rate=0.5)

# Utilizza CrossEntropyLoss con peso delle classi e Adam come ottimizzatore
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training del modello
num_epochs = 50
batch_size = 32

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_embeddings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()

# Liste per salvare le loss
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Valutazione del modello sul set di validazione
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_embeddings)
        val_loss = loss_fn(val_outputs, val_labels)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    # Salvare le loss per il grafico
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# Stampa il grafico delle loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Valutazione finale del modello sul set di test
model.eval()
with torch.no_grad():
    test_outputs = model(test_embeddings)
    _, predicted_labels = torch.max(test_outputs, 1)
    correct_predictions = (predicted_labels == test_labels).sum().item()
    total_samples = len(test_labels)
    accuracy = correct_predictions / total_samples * 100.0

print(f'Accuracy on test set: {accuracy:.2f}%')

# Stampa le previsioni sul set di test
class_names = label_encoder.classes_
predicted_class_names = [class_names[i] for i in predicted_labels.numpy()]


for i in range(20):  
    print(f"Example {i + 1}: Real Label - {class_names[test_labels[i]]}, Predicted Label - {predicted_class_names[i]}")
