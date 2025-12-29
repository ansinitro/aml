import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

# Configuration
DATA_FLAG = 'pneumoniamnist'
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
OUTPUT_DIR = 'results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"Loading {DATA_FLAG}...")
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)
    
    return train_dataset, val_dataset, test_dataset, info

def perform_eda(train_dataset, info):
    print("Performing EDA...")
    
    # 1. Class Distribution
    labels = train_dataset.labels
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels.flatten())
    plt.title("Class Distribution (Train)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(ticks=range(len(info['label'])), labels=info['label'].values())
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    # 2. Sample Images
    plt.figure(figsize=(10, 5))
    for i in range(10):
        img, label = train_dataset[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(info['label'][str(label.item())])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
    plt.close()

def flatten_data(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, targets in loader:
        X = data.view(data.size(0), -1).numpy()
        y = targets.numpy().flatten()
    return X, y

def week4_analysis(X, y, info):
    print("Performing Week 4 Analysis (PCA, t-SNE, Clustering)...")
    
    # Usage of a subset for plotting speed if dataset is huge, but PneumoniaMNIST is small (4708 samples)
    
    # 1. PCA for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=[info['label'][str(i)] for i in y], palette='viridis', alpha=0.7)
    plt.title("PCA Visualization (2 Components)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_visualization.png'))
    plt.close()
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X) # Can be slow, but ok for this size
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=[info['label'][str(i)] for i in y], palette='viridis', alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualization.png'))
    plt.close()
    
    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=len(info['label']), random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Confusion Matrix to see if clusters align with labels
    cm = confusion_matrix(y, clusters) # Note: Cluster IDs might not match Label IDs directly
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("K-Means Clusters vs True Labels")
    plt.ylabel("True Label")
    plt.xlabel("Cluster ID")
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_confusion.png'))
    plt.close()

def train_baseline_ml(X_train, y_train, X_test, y_test):
    print("Training Baseline ML (PCA + SVM)...")
    
    # PCA
    pca = PCA(n_components=0.95) # Retain 95% variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA reduced components: {pca.n_components_}")
    
    # SVM
    clf = SVC(probability=True, random_state=42)
    clf.fit(X_train_pca, y_train)
    
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline SVM Accuracy: {acc:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, 'baseline_results.txt'), 'w') as f:
        f.write(f"Baseline SVM Accuracy: {acc:.4f}\n")
        f.write(classification_report(y_test, y_pred))

def train_cnn(train_dataset, val_dataset, test_dataset, info):
    print("Training Deep Learning Model (CNN)...")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Simple CNN Architecture
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2), # 28 -> 14
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # 14 -> 7
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleCNN(num_classes=len(info['label'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
        
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"CNN Accuracy: {acc:.4f}")
    
    # Save Results
    with open(os.path.join(OUTPUT_DIR, 'cnn_results.txt'), 'w') as f:
        f.write(f"CNN Accuracy: {acc:.4f}\n")
        f.write(classification_report(all_targets, all_preds))
        
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=info['label'].values(), yticklabels=info['label'].values())
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_confusion_matrix.png'))
    plt.close()

def main():
    train_ds, val_ds, test_ds, info = load_data()
    
    perform_eda(train_ds, info)
    
    X_train, y_train = flatten_data(train_ds)
    X_test, y_test = flatten_data(test_ds)
    
    week4_analysis(X_train, y_train, info)
    
    train_baseline_ml(X_train, y_train, X_test, y_test)
    
    train_cnn(train_ds, val_ds, test_ds, info)
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
