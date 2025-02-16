import os
from torch.utils.data import DataLoader
import argparse
from classifiers import MnistClassifier
from utils import MnistDataset


def eval_models(datapath, batch_size, device, weights_dir, reports_dir):
    test_imgs_p = os.path.join(datapath, "t10k-images.idx3-ubyte")
    test_labels_p = os.path.join(datapath, "t10k-labels.idx1-ubyte")
    
    test_dataset = MnistDataset(test_imgs_p, test_labels_p)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    cnn_classifier = MnistClassifier("cnn", device=device)
    cnn_weights_path = os.path.join(weights_dir, "cnn_weights.pth")
    cnn_classifier.load(cnn_weights_path)
    print("Starting evaluation of CNN")
    cnn_report_path = os.path.join(reports_dir, "CNN_report.log")
    cnn_classifier.eval(test_loader, log_filename=cnn_report_path)
    
    fcnn_classifier = MnistClassifier("nn", device=device)
    fcnn_weights_path = os.path.join(weights_dir, "fcnn_weights.pth")
    fcnn_classifier.load(fcnn_weights_path)

    print("Starting evaluation of FCNN")
    fcnn_report_path = os.path.join(reports_dir, "FCNN_report.log")
    fcnn_classifier.eval(test_loader, log_filename=fcnn_report_path)

    rf_classifier = MnistClassifier("rf", device=device)
    rf_weights_path = os.path.join(weights_dir, "rf_weights.gz")
    rf_classifier.load(rf_weights_path)

    print("Starting evaluation of RF")
    rf_report_path = os.path.join(reports_dir, "RF_report.log")
    rf_classifier.eval(test_loader, log_filename=rf_report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for MNIST")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the folder with dataset (with default filenames).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for models.")
    parser.add_argument("--weights_dir",  type=str, required=True, help="Path to the folder for saved weights ('ll use default filenames).")
    parser.add_argument("--device", type=str, required=True, help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--reports_dir", type=str, default='./', help="Directory to save reports into.")
    args = parser.parse_args()
    
    eval_models(
        datapath=args.datapath,
        batch_size=args.batch_size,
        device=args.device,
        weights_dir=args.weights_dir,
        reports_dir=args.reports_dir)

