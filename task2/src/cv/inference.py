import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
# Define the same unique labels and target transform for test images
unique_labels = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

pred_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Should match the target size used during training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_animal(img_path, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = pred_transform(
        Image.open(img_path).convert('RGB')
        ).unsqueeze(0).to(device)
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
    
    model = model.to(device)
    
    model.load_state_dict(torch.load(weights_path,
                                     map_location=device,
                                     weights_only=True))
    model.eval()

    outputs = model(image)
    _, preds = torch.max(outputs, 1)

    return preds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference the animal classification model on the test set.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to image file.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the saved model checkpoint .")
    args = parser.parse_args()
    
    pred = predict_animal(img_path=args.img_path, weights_path=args.weights_path)
    pred_label = pred.tolist()[0]
    print(f"Label idx: {pred_label}, predicted name of animal: {unique_labels[pred_label]}")
