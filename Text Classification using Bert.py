import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

# Try to download stopwords, but provide a fallback
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load stopwords (fallback to an empty set if unavailable)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Remove non-alphanumeric characters
    text = text.strip()
    return text

# 🚀 Workaround: Use `split()` instead of `word_tokenize()`
def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()  # Replace NLTK tokenizer with basic split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load datasets
train_path = "C:\\Users\\bharg\\Downloads\\nlp-getting-started\\train.csv"
test_path = "C:\\Users\\bharg\\Downloads\\nlp-getting-started\\test.csv"
submission_path = "C:\\Users\\bharg\\Downloads\\nlp-getting-started\\submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Apply preprocessing
train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

print("✅ Text preprocessing completed successfully!")

# Load GloVe embeddings with error handling
try:
    print("🔄 Loading GloVe embeddings...")
    glove_vectors = api.load("glove-twitter-200")
    print("✅ GloVe embeddings loaded successfully.")
except Exception as e:
    print("❌ Error loading GloVe embeddings:", e)
    exit()

def get_glove_embedding(text):
    words = text.split()
    vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    if len(vectors) == 0:
        return np.zeros(200)
    return np.mean(vectors, axis=0)

train_df['glove_embedding'] = train_df['clean_text'].apply(get_glove_embedding)
test_df['glove_embedding'] = test_df['clean_text'].apply(get_glove_embedding)

# Convert embeddings to tensors
X = np.vstack(train_df['glove_embedding'].apply(lambda x: np.array(x)).values)
y = train_df['target'].values
X_test = np.vstack(test_df['glove_embedding'].apply(lambda x: np.array(x)).values)

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DisasterDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        if self.labels is not None:
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), torch.tensor(self.labels[idx])
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

# BERT Model
class DisasterBERT(nn.Module):
    def __init__(self):
        super(DisasterBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.sigmoid(logits).squeeze()

# Training Setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

def train_model(X, y):
    best_model = None
    best_f1 = 0
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"🚀 Training Fold {fold+1}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        train_dataset = DisasterDataset(train_df['clean_text'].iloc[train_index].values, y_train)
        val_dataset = DisasterDataset(train_df['clean_text'].iloc[val_index].values, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model = DisasterBERT().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.BCELoss()
        
        for epoch in range(3):
            model.train()
            total_loss = 0
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"📉 Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                outputs = model(input_ids, attention_mask).cpu().numpy()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())
        
        val_preds = np.array(val_preds) > 0.5
        f1 = f1_score(val_labels, val_preds)
        print(f"📊 Fold {fold+1} F1 Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    
    return best_model

# Train model
model = train_model(X, y)

# Generate Predictions
test_dataset = DisasterDataset(test_df['clean_text'].values)
test_loader = DataLoader(test_dataset, batch_size=16)

predictions = []
model.eval()
with torch.no_grad():
    for input_ids, attention_mask in test_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask).cpu().numpy()
        predictions.extend(outputs)

# Adjust thresholding for better performance
test_df['target'] = (np.array(predictions) > 0.4).astype(int)  

# Save submission
test_df[['id', 'target']].to_csv(submission_path, index=False)
print("✅ Submission file saved successfully!")
