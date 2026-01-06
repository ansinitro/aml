
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

def train_baseline_model(X_train, y_train, X_test, y_test):
    print("Training Baseline Model (Logistic Regression) with GridSearch...")
    
    # 1. Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 2. Hyperparameter Tuning using GridSearchCV
    # Sentiment140: 0=Negative, 4=Positive.
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_vec, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # 3. Evaluation
    y_pred = best_model.predict(X_test_vec)
    y_pred_proba = best_model.predict_proba(X_test_vec)[:, 1] # Probability of class 4
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=4)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results = {
        'model': best_model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'report': classification_report(y_test, y_pred),
        'feature_names': vectorizer.get_feature_names_out(),
        'coefficients': best_model.coef_[0]
    }
    
    print("Baseline Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(results['report'])
    
    return results

def train_naive_bayes_model(X_train, y_train, X_test, y_test):
    print("Training Naive Bayes Model...")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # MultinomialNB does not accept random_state
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    
    y_pred = nb.predict(X_test_vec)
    y_pred_proba = nb.predict_proba(X_test_vec)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=4)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results = {
        'model': nb,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'report': classification_report(y_test, y_pred)
    }
    
    print("Naive Bayes Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return results

def get_vader_scores(texts):
    print("Calculating VADER scores...")
    sia = SentimentIntensityAnalyzer()
    
    scores = []
    preds = []
    
    for text in texts:
        score = sia.polarity_scores(str(text))["compound"]
        scores.append(score)
        
        # Strict binary mapping for this dataset (0/4)
        # Using 0.05 threshold: > 0.05 is Positive (4), else Negative (0)
        # We treat neutral/weak sentiment as negative/0 for binary classification simplicity
        # or we could split at 0. Let's stick to VADER standard threshold.
        
        if score >= 0.05:
            preds.append(4) # Positive
        else:
            preds.append(0) # Negative (including neutral)

    return np.array(scores), np.array(preds)

def train_lstm_model(X_train, y_train, X_test, y_test, max_words=10000, max_len=100):
    print("Training Advanced Model (LSTM) with Optimization...")
    
    # Map labels to 0/1
    y_train_bin = np.where(y_train == 4, 1, 0)
    y_test_bin = np.where(y_test == 4, 1, 0)
    
    # 1. Tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # 2. Model Architecture
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2)) # recurrent_dropout can be slow on cuDNN
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 3. Callbacks (EarlyStopping)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
    ]
    
    # 4. Train
    print("Fitting LSTM...")
    history = model.fit(
        X_train_pad, y_train_bin, 
        epochs=5,
        batch_size=128, # Increased batch size for larger data
        validation_split=0.1, 
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluation
    y_pred_proba = model.predict(X_test_pad)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba)
    conf_matrix = confusion_matrix(y_test_bin, y_pred)
    
    results = {
        'model': model,
        'tokenizer': tokenizer,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'history': history
    }
    
    print(f"LSTM Model Accuracy: {accuracy:.4f}")
    print(f"LSTM F1-Score: {f1:.4f}")
    print(f"LSTM ROC-AUC: {roc_auc:.4f}")
    
    return results
