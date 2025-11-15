# 📩 SMS Spam Detection using Bag-of-Words & Machine Learning

This project builds a **spam classification model** using the **Bag-of-Words (BoW)** approach and traditional machine learning algorithms. It processes SMS messages, converts them into numerical features using text-preprocessing + BoW, and trains a classifier to detect **spam** vs **ham** messages.

---

## 🚀 Project Overview

* Clean and preprocess raw SMS text
* Tokenize, lemmatize, and remove noise
* Convert text into numerical vectors using **CountVectorizer**
* Train a machine learning model to classify messages
* Evaluate the model using accuracy, confusion matrix, etc.

---

## 📂 Dataset

The dataset used is the classic **SMS Spam Collection** dataset.
Each row contains:

* **label** → `ham` or `spam`
* **message** → the SMS text

---

## 🧼 Text Preprocessing Steps

1. Remove non-alphabetical characters
2. Convert to lowercase
3. Tokenize the text
4. Apply lemmatization
5. Join tokens back to cleaned strings
6. Store all processed messages in a final corpus

---

## 🔧 Feature Extraction

The cleaned corpus is passed through:

### **Bag of Words (CountVectorizer)**

* Converts text into a sparse matrix
* Each feature represents a unique word
* Each value represents the word count in a message

---

## 🤖 Model Used

You can plug in any ML model, but commonly used are:

* **Naive Bayes (MultinomialNB)**
* **Logistic Regression**
* **SVM** (Optional)

---

## 📊 Evaluation

Standard metrics used:

* **Accuracy score**
* **Confusion Matrix**
* **Precision, Recall, F1 Score**

---

## 🛠 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 🧪 How to Run

1. Install required libraries:

   ```bash
   pip install pandas numpy nltk scikit-learn
   ```
2. Download NLTK resources:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
3. Open the notebook:

   ```
   bag_of_words.ipynb
   ```
4. Run all cells in order.

---

## 📈 Results

The model achieves high accuracy in detecting spam messages using basic NLP + ML techniques.
Bag-of-Words proves to be a simple yet effective feature extraction method.

---
