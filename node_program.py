import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import requests
import threading
import time
import json

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Global configuration for progress reporting
SERVER_URL = "http://proj.ziqfm.com"  # central server URL
NODE_ID = os.environ.get("NODE_ID", "node_1")  # unique node id

# --- Helper functions for server communication ---

def request_job():
    try:
        response = requests.get(f"{SERVER_URL}/api/get_job", params={"node_id": NODE_ID})
        if response.status_code == 200:
            job = response.json()
            print("Received job from server:", job)
            return job
        else:
            print("Failed to get job, status code:", response.status_code)
    except Exception as e:
        print("Error requesting job:", e)
    return None

def download_checkpoint(checkpoint_url):
    try:
        response = requests.get(checkpoint_url)
        if response.status_code == 200:
            checkpoint_path = "checkpoint.pth"
            with open(checkpoint_path, "wb") as f:
                f.write(response.content)
            print("Checkpoint downloaded.")
            return checkpoint_path
    except Exception as e:
        print("Error downloading checkpoint:", e)
    return None

def upload_snapshot(epoch, metrics, checkpoint_path, job_id, model_name):
    try:
        files = {}
        if os.path.exists(checkpoint_path):
            files['checkpoint'] = open(checkpoint_path, 'rb')
        data = {
            "node_id": NODE_ID,
            "job_id": job_id,
            "model_name": model_name,
            "epoch": epoch,
            "metrics": json.dumps(metrics)
        }
        response = requests.post(f"{SERVER_URL}/api/upload_snapshot", data=data, files=files)
        if response.status_code == 200:
            print("Snapshot uploaded successfully for epoch", epoch)
        else:
            print("Failed to upload snapshot, status code:", response.status_code)
    except Exception as e:
        print("Error uploading snapshot:", e)

def progress_reporter(stop_event, progress_info_func, poll_interval=60):
    while not stop_event.is_set():
        progress_info = progress_info_func()
        try:
            response = requests.post(f"{SERVER_URL}/api/update_progress", json=progress_info)
            if response.status_code == 200:
                print("Progress updated:", progress_info)
            else:
                print("Failed to update progress, status code:", response.status_code)
        except Exception as e:
            print("Error updating progress:", e)
        time.sleep(poll_interval)

# --- Dataset class (same as before) ---

class PneumoniaDataset(Dataset):
    def __init__(self, images_path, metadata_path, masks_path=None, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.has_masks = masks_path is not None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_path, f"{row['patientId']}.png")
        label = row['Target']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        if self.has_masks:
            mask_path = os.path.join(self.masks_path, f"{row['patientId']}.png")
            try:
                mask = Image.open(mask_path).convert('L')
                mask = transforms.ToTensor()(mask)
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask = torch.zeros((1, 224, 224))
            return image, mask, label
        return image, label

# --- Model builder (unchanged) ---

def get_model(model_name, num_classes=2):
    from torchvision.models import (
        ResNet50_Weights,
        DenseNet121_Weights,
        GoogLeNet_Weights,
        EfficientNet_B0_Weights,
        VGG16_Weights,
        MobileNet_V2_Weights
    )
    models_dict = {
        'resnet50': models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        'densenet121': models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
        'googlenet': models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1),
        'efficientnet': models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
        'vgg16': models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        'mobilenet': models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    }
    model = models_dict.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} is not supported.")
    if model_name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'googlenet':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --- Plotting functions (unchanged) ---

def plot_training_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes, model_name, classifier_name, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name} with {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_scores, model_name, classifier_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {model_name} with {classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

# --- Training functions, now with checkpoint saving and progress upload ---

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cuda', start_epoch=0, job_id="unknown_job"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': []}
    global current_epoch
    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch + 1  # update global progress
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            if len(batch) == 3:
                images, masks, labels = batch
                images, labels = images.to(device), labels.to(device)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        avg_train_loss = train_loss / total
        train_acc = 100. * correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, masks, labels = batch
                else:
                    images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        avg_val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        history.setdefault('val_loss', []).append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # After each epoch, save checkpoint and upload snapshot
        metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        }
        torch.save(model.state_dict(), "checkpoint.pth")
        upload_snapshot(epoch+1, metrics, "checkpoint.pth", job_id, model.__class__.__name__)
    plot_training_history(history, model.__class__.__name__)
    return model

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, masks, batch_labels = batch
            else:
                images, batch_labels = batch
            images = images.to(device)
            if hasattr(model, 'fc'):
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                features_batch = torch.flatten(x, 1)
            elif hasattr(model, 'classifier'):
                x = model.features(images)
                x = model.avgpool(x)
                features_batch = torch.flatten(x, 1)
            else:
                features_batch = model(images)
            features.append(features_batch.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)

def train_evaluate_classifiers(train_features, train_labels, test_features, test_labels, device='cpu'):
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    results = {}
    for name, clf in classifiers.items():
        print(f'\nTraining {name} classifier...')
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        probabilities = clf.predict_proba(test_features)[:,1] if hasattr(clf, "predict_proba") else None
        report = classification_report(test_labels, predictions, output_dict=True)
        cm = confusion_matrix(test_labels, predictions)
        results[name] = {
            'classifier': clf,
            'predictions': predictions,
            'probabilities': probabilities,
            'report': report,
            'confusion_matrix': cm
        }
        print(f'Classification Report for {name}:\n')
        print(classification_report(test_labels, predictions))
        acc = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        try:
            roc_auc = roc_auc_score(test_labels, probabilities)
        except:
            roc_auc = 'N/A'
        mcc = matthews_corrcoef(test_labels, predictions)
        print(f'Accuracy: {acc:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC-AUC: {roc_auc}')
        print(f'Matthews Correlation Coefficient: {mcc:.4f}')
        plot_confusion_matrix(cm, classes=['Negative', 'Positive'], model_name='Model', classifier_name=name)
        if probabilities is not None and roc_auc != 'N/A':
            plot_roc_curve(test_labels, probabilities, model_name='Model', classifier_name=name)
    return results

def hyperparameter_tuning(model_name, train_loader, val_loader, learning_rates, num_epochs, device):
    tuning_results = {}
    for lr in learning_rates:
        print(f'\nTraining {model_name} with learning rate: {lr}')
        model = get_model(model_name)
        trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs,
                                    learning_rate=lr, device=device)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, masks, labels = batch
                else:
                    images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        tuning_results[lr] = val_acc
        print(f'Validation Accuracy: {val_acc:.2f}%')
    plt.figure(figsize=(8,6))
    lrs = list(tuning_results.keys())
    accs = list(tuning_results.values())
    plt.plot(lrs, accs, marker='o')
    plt.title(f'Learning Rate Tuning for {model_name}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy (%)')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    best_lr = max(tuning_results, key=tuning_results.get)
    print(f'Best Learning Rate for {model_name}: {best_lr} with Validation Accuracy: {tuning_results[best_lr]:.2f}%')
    return best_lr

# --- Main program ---

def main():
    # Request job assignment from server
    job = request_job()
    if job is None or job.get("message"):
        print("No job assigned. Exiting.")
        return
    job_id = job.get("job_id", "unknown_job")
    model_name = job.get("model_name", "resnet50")
    perform_tuning = job.get("perform_tuning", False)
    resume = job.get("resume", False)
    checkpoint_url = job.get("checkpoint_url", None)
    current_epoch_saved = job.get("current_epoch", 0)

    # Set dataset paths (the dataset was extracted by the Dockerfile)
    train_img_path = "/app/Training/Images"
    train_mask_path = "/app/Training/Masks"
    train_metadata = "/app/stage2_train_metadata.csv"
    test_metadata = "/app/stage2_test_metadata.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PneumoniaDataset(train_img_path, train_metadata, train_mask_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,          # Batch size set to 64
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    if perform_tuning:
        learning_rates = [1e-4, 1e-3, 1e-2]
        num_epochs = 50
    else:
        learning_rate = 0.001
        num_epochs = 100

    model = get_model(model_name)
    start_epoch = 0
    if resume and checkpoint_url:
        cp_path = download_checkpoint(checkpoint_url)
        if cp_path:
            model.load_state_dict(torch.load(cp_path))
            start_epoch = current_epoch_saved
            print(f"Resuming training from epoch {start_epoch+1}")

    # Start a background thread for progress reporting
    stop_event = threading.Event()
    def get_progress_info():
        return {
            "node_id": NODE_ID,
            "model_name": model_name,
            "current_epoch": globals().get("current_epoch", 0),
            "job_id": job_id
        }
    reporter_thread = threading.Thread(target=progress_reporter, args=(stop_event, get_progress_info, 60))
    reporter_thread.start()

    if perform_tuning:
        best_lr = hyperparameter_tuning(model_name, train_loader, val_loader, learning_rates, num_epochs, device)
        lr = best_lr
    else:
        lr = learning_rate

    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs,
                                learning_rate=lr, device=device, start_epoch=start_epoch, job_id=job_id)
    torch.save(trained_model.state_dict(), f'{model_name}_pneumonia.pth')
    print(f"Saved {model_name} model as {model_name}_pneumonia.pth")
    stop_event.set()
    reporter_thread.join()

    print(f'Extracting features for {model_name}...')
    train_features, train_labels = extract_features(trained_model, train_loader, device)
    val_features, val_labels = extract_features(trained_model, val_loader, device)
    print(f'Training and evaluating classifiers for {model_name}...')
    train_evaluate_classifiers(train_features, train_labels, val_features, val_labels, device=device)

if __name__ == "__main__":
    main()
