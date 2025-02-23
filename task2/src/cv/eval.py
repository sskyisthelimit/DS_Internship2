import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report
from utils import AnimalDataset  # Ensure this is accessible

# Define the same unique labels and target transform for test images
unique_labels = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Should match the target size used during training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_model(test_csv, checkpoint_path, batch_size):
    # Load test set CSV file (which has columns Filepath, Label, Label_idx)
    test_df = pd.read_csv(test_csv)
    
    test_dataset = AnimalDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build the same model architecture
    model = models.efficientnet_v2_m(weights='DEFAULT')
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
    
    # Load the saved model weights
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device,
                                     weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("Test Set Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=unique_labels))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the animal classification model on the test set.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the CSV file containing test set filepaths and labels (e.g., test_set_filepaths.csv).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model checkpoint (e.g., animals_classification_model_checkpoint.pth).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()
    
    evaluate_model(test_csv=args.test_csv, checkpoint_path=args.checkpoint, batch_size=args.batch_size)
