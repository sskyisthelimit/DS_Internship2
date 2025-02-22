import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report
from src.cv.utils import AnimalDataset  # Make sure this accepts a DataFrame and a transform

# Label mapping and unique labels
label_mapping = {
    'cane': 'dog',
    'cavallo': 'horse',
    'elefante': 'elephant',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'gatto': 'cat',
    'mucca': 'cow',
    'pecora': 'sheep',
    'ragno': 'spider',
    'scoiattolo': 'squirrel'
}
unique_labels = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

seed_everything()

def train(dataset_dir, batch_size, target_size,
          test_set_paths,
          ckpt_save_path,
          num_epochs):
    
    def convert_path_to_df(dataset):
        image_dir = Path(dataset)
        filepaths = list(image_dir.glob('**/*.JPG')) + list(image_dir.glob('**/*.jpg')) + \
                    list(image_dir.glob('**/*.jpeg')) + list(image_dir.glob('**/*.PNG'))
        # Assume the parent folder name is the label
        labels = [Path(fp).parent.name for fp in filepaths]
        image_df = pd.DataFrame({'Filepath': list(map(str, filepaths)), 'Label': labels})
        return image_df

    image_df = convert_path_to_df(dataset_dir)
    # Map labels to common names and create numeric labels
    image_df['Label'] = image_df['Label'].map(label_mapping)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    image_df['Label_idx'] = image_df['Label'].map(label_to_idx)

    train_df, temp_df = train_test_split(image_df, test_size=0.3, shuffle=True, 
                                           random_state=42, stratify=image_df['Label_idx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.66, shuffle=True, 
                                         random_state=42, stratify=temp_df['Label_idx'])

    # Save test filepaths and labels for later evaluation
    test_df.to_csv(test_set_paths, index=False)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders for train, validation, and (optionally) test
    train_dataset = AnimalDataset(train_df, transform=train_transform)
    val_dataset = AnimalDataset(val_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model using EfficientNetV2_M and modify classifier
    model = models.efficientnet_v2_m(weights='DEFAULT')
    # Freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.45),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.45),
        nn.Linear(256, len(unique_labels))
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    DEVICE = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)

    best_val_loss = np.inf
    patience = 5
    epochs_without_improvement = 0

    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        
        # Evaluate on validation set
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=unique_labels))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train an animal classification model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset images arranged in subfolders.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224], help="Target image size as two integers, e.g., 224 224.")
    parser.add_argument("--test_set_paths", type=str, default="./test_set_filepaths.csv", help="Path to test set image paths .csv")
    parser.add_argument("--ckpt_save_path", type=str, default="./animals_checkpoint.pth", help="Path to save checkpoint")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs for training")

    args = parser.parse_args()
    
    train(dataset_dir=args.dataset_dir, batch_size=args.batch_size,
          target_size=tuple(args.target_size),
          test_set_paths=args.test_set_paths,
          ckpt_save_path=args.ckpt_save_path,
          num_epochs=args.num_epochs,)
