import abc
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class MnistClassifierInterface(abc.ABC):
    @abc.abstractmethod
    def train(self, loader, epochs=7):
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def save(self, filename):
        pass

    @abc.abstractmethod
    def load(self, filename):
        pass


class RFMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, loader, epochs=1):
        X_train, y_train = [], []
        for images, labels in loader:
            X_train.extend(images.view(images.shape[0], -1).numpy())
            y_train.extend(labels.numpy())
        self.model.fit(X_train, y_train)

    def eval(self, loader, log_file="RF_report.log"):
        y_true, y_pred = [], []
        pbar = tqdm(loader, desc="RF Evaluation", unit="batch")
        for images, labels in pbar:
            preds = self.predict(images)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
        report = classification_report(y_true, y_pred, digits=4)
        with open(log_file, "w") as f:
            f.write(report)
        
        print("RF classification report")
        print(report)

    def predict(self, input):
        input = input.view(input.shape[0], -1).numpy()
        return self.model.predict(input)
    
    def save(self, filename):
        joblib.dump(self.model, filename=filename)

    def load(self, filename):
        self.model = joblib.load(filename)


class NNMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_classes=10, device="cpu", input_shape=(28, 28)):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        ).to(device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, loader, epochs=7):
        self.model.train()
        for epoch in range(epochs):
            total_loss, correct_n = 0, 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct_n += (pred.argmax(1) == labels).sum().item()

                pbar.set_postfix(loss=total_loss/len(loader),
                                 accuracy=correct_n/len(loader.dataset)*100)
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, "
                  f"Accuracy: {correct_n/len(loader.dataset)*100:.2f}%")

    def eval(self, loader, log_filename="FCNN_report.log"):
        self.model.eval()
        y_true, y_pred = [], []
        pbar = tqdm(loader, desc="FCNN Evaluation", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        report = classification_report(y_true, y_pred, digits=4)
        with open(log_filename, "w") as f:
            f.write(report)
        print("FCNN classification report")
        print(report)

    def predict(self, input):
        self.model.eval()
        input = input.to(self.device)
        with torch.no_grad():
            pred = self.model(input)
        return pred.argmax(1)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename,
                                              map_location=self.device))


class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self, in_channels=1, n_classes=10, device="cpu"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reducing 28x28 → 14x14
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        ).to(device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, loader, epochs=7):
        self.model.train()
        for epoch in range(epochs):
            total_loss, correct_n = 0, 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct_n += (pred.argmax(1) == labels).sum().item()

                pbar.set_postfix(loss=total_loss/len(loader),
                                 accuracy=correct_n/len(loader.dataset)*100)
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, "
                  f"Accuracy: {correct_n/len(loader.dataset)*100:.2f}%")
    
    def eval(self, loader, log_filename="CNN_report.log"):
        self.model.eval()
        y_true, y_pred = [], []
        pbar = tqdm(loader, desc="CNN Evaluation", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        report = classification_report(y_true, y_pred, digits=4)
        with open(log_filename, "w") as f:
            f.write(report)
        
        print("CNN classification report")
        print(report)
            
    def predict(self, input):
        self.model.eval()
        input = input.to(self.device)
        with torch.no_grad():
            pred = self.model(input)
        return pred.argmax(1)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename,
                                              map_location=self.device))


class MnistClassifier:
    def __init__(self, algorithm, device="cpu"):
        if algorithm == "rf":
            self.model = RFMnistClassifier()
        elif algorithm == "nn":
            self.model = NNMnistClassifier(device=device)
        elif algorithm == "cnn":
            self.model = CNNMnistClassifier(device=device)
        else:
            raise ValueError("Invalid algorithm. Choose 'cnn', 'nn', or 'rf'.")

    def train(self, loader, epochs=7):
        self.model.train(loader, epochs)

    def predict(self, input):
        return self.model.predict(input)
