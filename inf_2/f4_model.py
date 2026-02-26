import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Face_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8*8*256, 512)

        # Heads
        self.landmark_head = nn.Linear(512, 10)
        self.emotion_head = nn.Linear(512, 5)
        self.race_head = nn.Linear(512, 3)
        self.gender_head = nn.Linear(512, 3)
        self.age_head = nn.Linear(512, 5)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))

        landmarks = self.landmark_head(x)
        emotion = self.emotion_head(x)
        race = self.race_head(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return landmarks, emotion, race, gender, age


class Face_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(512, 512)

        self.landmark_head = nn.Linear(512, 10)
        self.emotion_head = nn.Linear(512, 5)
        self.race_head = nn.Linear(512, 3)
        self.gender_head = nn.Linear(512, 3)
        self.age_head = nn.Linear(512, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        landmarks = self.landmark_head(x)
        emotion = self.emotion_head(x)
        race = self.race_head(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return landmarks, emotion, race, gender, age


class Face_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.emotion_head = nn.Linear(512, 5)
        self.race_head = nn.Linear(512, 3)
        self.gender_head = nn.Linear(512, 3)
        self.age_head = nn.Linear(512, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        landmarks = torch.sigmoid(self.landmark_head(x))
        emotion = self.emotion_head(x)
        race = self.race_head(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return landmarks, emotion, race, gender, age


def initialize_model(arch,device='cuda', lr=1e-3):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arch()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer#, device

def initialize_model2(arch,device='cuda', lr=1e-3, optimizer=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arch()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer

    return model, optimizer#, device


def train_model(model, dataloaders, optimizer, losses, name, device='cuda', 
                num_epochs=10, loss_weights=None, writer=None):
    writer = SummaryWriter(log_dir=f'runs/{name}')
    if loss_weights is None: loss_weights = {k: 1.0 for k in losses.keys()}
    
    history = {'train': {k: [] for k in losses.keys()},
               'val': {k: [] for k in losses.keys()}} if 'val' in dataloaders else {'train': {k: [] for k in losses.keys()}}
    
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val'] if 'val' in dataloaders else ['train']:
            if phase == 'train': model.train()
            else: model.eval()
            
            running_loss = {k: 0.0 for k in losses.keys()}
            
            loader = dataloaders[phase]
            
            for images, labels in tqdm(loader, desc=f"{phase}"):
                images = images.to(device)
                
                gt = {}
                for key in losses.keys():
                    gt[key] = labels[key].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    preds = model(images)
                    task_losses = {}
                    total_loss = 0.0
                    for i, key in enumerate(losses.keys()):
                        target = gt[key]
                        # flatten landmarks
                        if key == 'landmarks':
                            target = target.view(target.size(0), -1)  # [B, 5, 2] -> [B, 10]
                        task_loss = losses[key](preds[i], target) * loss_weights.get(key, 1.0)
                        task_losses[key] = task_loss
                        total_loss += task_loss
                    
                    if phase=='train':
                        total_loss.backward()
                        optimizer.step()
                
                for key in running_loss.keys():
                    running_loss[key] += task_losses[key].item()
            
            # average over batches
            num_batches = len(loader)
            avg_loss = {k: running_loss[k]/num_batches for k in running_loss.keys()}
            for k in avg_loss.keys():
                history[phase][k].append(avg_loss[k])
            
            if writer:
                for k in avg_loss.keys():
                    writer.add_scalar(f"{phase}_loss/{k}", avg_loss[k], epoch)
            
            print(f"{phase} Losses: " + ", ".join([f"{k}: {avg_loss[k]:.4f}" for k in avg_loss.keys()]))
    
    return history



def compute_metrics(preds, targets):
    metrics = {}

    landmarks_pred = preds['landmarks'].cpu().numpy()
    landmarks_true = targets['landmarks'].cpu().numpy()
    metrics['landmarks_mse'] = mean_squared_error(
        landmarks_true.reshape(landmarks_true.shape[0], -1),
        landmarks_pred.reshape(landmarks_pred.shape[0], -1)
    )

    for i, task in enumerate(['emotion', 'race', 'gender', 'age']):
        y_pred = torch.argmax(preds[task], dim=1).cpu().numpy()
        y_true = targets[task].cpu().numpy()
        print(f"{task} - y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
        metrics[f'{task}_acc'] = accuracy_score(y_true, y_pred)
        metrics[f'{task}_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics[f'{task}_precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics[f'{task}_recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics[f'{task}_confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics



def predict(model, image_path, device='cuda'):
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # Assuming bounding box is full image; adjust if using face detection
    x1, y1, x2, y2 = 0, 0, W, H
    image_crop = image[y1:y2, x1:x2]

    # Resize to model input
    image_resized = cv2.resize(image_crop, (128, 128))
    image_tensor = torch.tensor(image_resized.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)/255.0
    image_tensor = image_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        landmarks_pred, emotion_pred, race_pred, gender_pred, age_pred = model(image_tensor)

    # Convert outputs
    landmarks_pred = landmarks_pred.cpu().numpy().reshape(-1,2)
    # scale back to original crop size
    landmarks_pred[:,0] *= (x2-x1)
    landmarks_pred[:,1] *= (y2-y1)

    # Categorical predictions
    emotion_label = torch.argmax(emotion_pred, dim=1).item()
    race_label = torch.argmax(race_pred, dim=1).item()
    gender_label = torch.argmax(gender_pred, dim=1).item()
    age_label = torch.argmax(age_pred, dim=1).item()

    return {
        'landmarks': landmarks_pred,
        'emotion': emotion_label,
        'race': race_label,
        'gender': gender_label,
        'age': age_label
    }


def show_prediction(image_path, prediction):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    
    # Draw landmarks
    for x, y in prediction['landmarks']:
        plt.scatter(x, y, c='r', s=30)
    
    plt.title(f"Emotion: {prediction['emotion']}, Race: {prediction['race']}, "
              f"Gender: {prediction['gender']}, Age: {prediction['age']}")
    plt.axis('off')
    plt.show()