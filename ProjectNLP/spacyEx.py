from collections import Counter
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.lang.it.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

# Carica i tre dataset per addestramento, test e validazione separatamente
training_dataset = load_dataset("itacasehold/itacasehold", split = 'train')
test_dataset = load_dataset("itacasehold/itacasehold", split='test')
validation_dataset = load_dataset("itacasehold/itacasehold", split='validation')

# Estrai documenti e etichette da ciascun dataset
training_documents = training_dataset["summary"]
training_labels = training_dataset["materia"]

# Conta le occorrenze di ciascuna etichetta nel dataset
label_counts = Counter(training_labels)

# Trova etichette che compaiono almeno 5 volte
common_labels = [label for label, count in label_counts.items() if count >= 7]

# Filtra i documenti e le etichette mantenendo solo quelli con etichette comuni
filtered_indices = [i for i, label in enumerate(training_labels) if label in common_labels]
training_documents = [training_documents[i] for i in filtered_indices]
training_labels = [training_labels[i] for i in filtered_indices]

test_documents = test_dataset["summary"]
test_labels = test_dataset["materia"]

# Filtra i documenti e le etichette mantenendo solo quelli con etichette comuni
filtered_indices = [i for i, label in enumerate(test_labels) if label in common_labels]
test_documents = [test_documents[i] for i in filtered_indices]
test_labels = [test_labels[i] for i in filtered_indices]

validation_documents = validation_dataset["summary"]
validation_labels = validation_dataset["materia"]

# Filtra i documenti e le etichette mantenendo solo quelli con etichette comuni
filtered_indices = [i for i, label in enumerate(validation_labels) if label in common_labels]
validation_documents = [validation_documents[i] for i in filtered_indices]
validation_labels = [validation_labels[i] for i in filtered_indices]


# Rimuovi la punteggiatura, lemmatizza e rimuovi le stop words dal dataset di Spacy
nlp = spacy.load('it_core_news_lg')

training_documents = [
    " ".join([token.lemma_.lower() for token in nlp(doc) if not token.is_punct and token.text.lower() not in STOP_WORDS])
    for doc in training_documents
]

test_documents = [
    " ".join([token.lemma_.lower() for token in nlp(doc) if not token.is_punct and token.text.lower() not in STOP_WORDS])
    for doc in test_documents
]

validation_documents = [
    " ".join([token.lemma_.lower() for token in nlp(doc) if not token.is_punct and token.text.lower() not in STOP_WORDS])
    for doc in validation_documents
]

# Usa il TF-IDF per ottenere le rappresentazioni vettoriali
vectorizer = TfidfVectorizer()
training_tfidf_matrix = vectorizer.fit_transform(training_documents)
test_tfidf_matrix = vectorizer.transform(test_documents)
validation_tfidf_matrix = vectorizer.transform(validation_documents)

# Converte l'array numpy TF-IDF in un tensore PyTorch
training_embeddings_tensor = torch.tensor(training_tfidf_matrix.toarray(), dtype=torch.float32)
test_embeddings_tensor = torch.tensor(test_tfidf_matrix.toarray(), dtype=torch.float32)
validation_embeddings_tensor = torch.tensor(validation_tfidf_matrix.toarray(), dtype=torch.float32)

# Tratta labels come una lista di stringhe
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
test_labels_encoded = label_encoder.transform(test_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)

training_labels_tensor = torch.tensor(training_labels_encoded, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)
validation_labels_tensor = torch.tensor(validation_labels_encoded, dtype=torch.long)

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
input_size = training_tfidf_matrix.shape[1]  # Dimensione con TF-IDF
hidden_size = 64
output_size = len(set(training_labels))

# Inizializza il modello
model = SimpleClassifier(input_size, hidden_size, output_size, dropout_rate=0.5)

# Utilizza CrossEntropyLoss con peso delle classi e Adam come ottimizzatore
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training del modello
num_epochs = 150
batch_size = 128

train_dataset = TensorDataset(training_embeddings_tensor, training_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(validation_embeddings_tensor, validation_labels_tensor)
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
        val_outputs = model(validation_embeddings_tensor)
        val_loss = loss_fn(val_outputs, validation_labels_tensor)
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
    test_outputs = model(test_embeddings_tensor)
    _, predicted_labels = torch.max(test_outputs, 1)
    correct_predictions = (predicted_labels == test_labels_tensor).sum().item()
    total_samples = len(test_labels_tensor)
    accuracy = correct_predictions / total_samples * 100.0

print(f'Accuracy on test set: {accuracy:.2f}%')

# Stampa le previsioni sul set di test
class_names = label_encoder.classes_
predicted_class_names = [class_names[i] for i in predicted_labels.numpy()]

for i in range(20):  
    print(f"Example {i + 1}: Real Label - {class_names[test_labels_encoded[i]]}, Predicted Label - {predicted_class_names[i]}")
