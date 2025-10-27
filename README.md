# ANN2_Project:
This is a repo for the Project that I have done. It also contains the 
    - Project Report, 
    - Source Code + Documentation.
To install all requirements
pip install -r requirements.txt

# NLP Disaster Tweet Classification

## 📌 Project Overview
This project aims to classify tweets as either related to real disasters (1) or not (0). It utilizes both GloVe word embeddings and BERT for text processing and classification. The dataset used is from the Kaggle competition "Real or Not? NLP with Disaster Tweets."

## 📂 Project Structure
- `train.csv`: Training dataset containing tweets and labels.
- `test.csv`: Test dataset without labels (to be predicted).
- `submission.csv`: The generated submission file with predictions.
- `Text Classification using Bert.py`: The main script implementing data preprocessing, embedding generation, and BERT model training.

## 🛠 Dependencies & Installation
Ensure you have Python installed and install the required dependencies:
```bash
pip install numpy pandas torch transformers nltk gensim scikit-learn
```
If `stopwords` are missing, download them manually:
```python
import nltk
nltk.download('stopwords')
```

## 📊 Data Preprocessing
- Convert text to lowercase.
- Remove URLs and non-alphanumeric characters.
- Tokenization (using `split()` instead of NLTK tokenizer for simplicity).
- Remove stopwords.
- Generate GloVe embeddings.

## 🔢 Embedding Generation
- Load `glove-twitter-200` from `gensim`.
- Compute word embeddings by averaging available word vectors.

## 🤖 Model Architecture
### 1️⃣ **GloVe Embeddings-Based Model**
- Converts tweets into 200-dimensional vectors using GloVe.

### 2️⃣ **BERT-based Model**
- Uses a `BertTokenizer` for tokenizing text.
- A `BertModel` followed by a linear layer and sigmoid activation for binary classification.

## 🏋️‍♂️ Model Training
- Uses **5-Fold Stratified Cross-Validation**.
- Trains for **3 epochs per fold** with **AdamW optimizer**.
- Uses **Binary Cross-Entropy Loss**.
- Tracks **F1 Score** for performance evaluation.

## 🚀 Running the Script
To train the model and generate predictions:
```bash
python Text Classification using Bert.py
```
This will:
1. Preprocess the text.
2. Generate GloVe embeddings.
3. Train the BERT model using cross-validation.
4. Predict on test data.
5. Save the final submission file as `submission.csv`.

## 📈 Results & Submission
- The model outputs predictions as probabilities.
- A threshold of **0.4** is applied to determine class labels.
- The final predictions are stored in `submission.csv`.

## 📌 Notes & Enhancements
- The script automatically selects `cuda` if available for GPU acceleration.
- Future enhancements can include fine-tuning the BERT model further and experimenting with different thresholds for classification.

## 🏆 Acknowledgments
- **Kaggle** for the dataset.
- **Hugging Face Transformers** for the BERT implementation.
- **Gensim** for pre-trained word embeddings.

