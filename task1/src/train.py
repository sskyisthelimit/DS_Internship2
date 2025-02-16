import os
import argparse
from torch.utils.data import DataLoader
from classifiers import MnistClassifier
from utils import MnistDataset


def train_models(datapath, batch_size, device, weights_save_dir):
    train_imgs_p = os.path.join(datapath, "train-images.idx3-ubyte")
    train_labels_p = os.path.join(datapath, "train-labels.idx1-ubyte")
    
    train_dataset = MnistDataset(train_imgs_p, train_labels_p)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    cnn_classifier = MnistClassifier("cnn", device=device)
    print("Starting to train CNN")
    cnn_classifier.train(train_loader, epochs=10)
    cnn_weights_path = os.path.join(weights_save_dir, "cnn_weights.pth")
    cnn_classifier.model.save(cnn_weights_path)
    
    fcnn_classifier = MnistClassifier("nn", device=device)
    print("Starting to train FCNN")
    fcnn_classifier.train(train_loader, epochs=10)
    fcnn_weights_path = os.path.join(weights_save_dir, "fcnn_weights.pth")
    fcnn_classifier.model.save(fcnn_weights_path)

    rf_classifier = MnistClassifier("rf", device=device)
    print("Starting to train RF")
    rf_classifier.train(train_loader)
    rf_weights_path = os.path.join(weights_save_dir, "rf_weights.gz")
    rf_classifier.model.save(rf_weights_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for MNIST")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the folder with dataset (with default filenames).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for models.")
    parser.add_argument("--weights_save_dir",  type=str, required=True, help="Path to the folder for weight ('ll use default filenames).")
    parser.add_argument("--device", type=str, required=True, help="Device to use (e.g., 'cuda:0' or 'cpu').")
    args = parser.parse_args()
    
    train_models(
        datapath=args.datapath,
        batch_size=args.batch_size,
        device=args.device,
        weights_save_dir=args.weights_save_dir)

