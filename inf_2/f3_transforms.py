import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, df, image_size=128, training=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.training = training

        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            # Use Affine with rotate to avoid Rotate duplicating keypoints
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(image_size, image_size)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.no_aug = A.Compose([
            A.Resize(image_size, image_size)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.left_right_pairs = [(0, 1), (3, 4)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {row.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = row.bbox
        w, h = x2 - x1, y2 - y1
        # print(f"Original bbox: {row.bbox}")
        if self.training:
            expand_ratio = np.random.uniform(0.05, 0.15)
            x1 = max(0, x1 - int(w * expand_ratio))
            y1 = max(0, y1 - int(h * expand_ratio))
            x2 = min(image.shape[1], x2 + int(w * expand_ratio))
            y2 = min(image.shape[0], y2 + int(h * expand_ratio))


        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # ensure bbox is valid and within image bounds (avoid zero-sized crops)
        H, W = image.shape[:2]
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))
        image = image[y1:y2, x1:x2]

        landmarks = np.array(row['landmarks_manual'], dtype=np.float32)
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1
        # sanitize numeric issues and clip to crop bounds before Albumentations
        h0, w0 = image.shape[:2]
        landmarks[:, :2] = np.nan_to_num(landmarks[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, w0 - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, h0 - 1)

        if self.training:
            aug = self.augment(image=image, keypoints=landmarks.tolist())
            image = aug['image']
            landmarks = np.array(aug['keypoints'], dtype=np.float32)

        else:
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0.0, w0 - 1e-4)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0.0, h0 - 1e-4)
            aug = self.no_aug(image=image, keypoints=landmarks.tolist())
            image = aug['image']
            landmarks = np.array(aug['keypoints'], dtype=np.float32)


        h_crop, w_crop = image.shape[:2]
        landmarks[:, 0] /= w_crop
        landmarks[:, 1] /= h_crop

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        labels = {
            'emotion': torch.tensor(row.emotion, dtype=torch.long),
            'race': torch.tensor(row.race, dtype=torch.long),
            'gender': torch.tensor(row.gender, dtype=torch.long),
            'age': torch.tensor(row.age, dtype=torch.long),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32)
        }

        return image, labels



def show_image_with_landmarks(image_tensor, landmarks, title=""):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    h, w = image.shape[:2]
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h

    plt.imshow(image)
    plt.scatter(landmarks_px[:, 0], landmarks_px[:, 1], c='r', s=40)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    df = pd.read_pickle('raf_slightly_mod.pkl')
    dataset = FaceDataset(df, image_size=128, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Take one batch
    images, labels = next(iter(dataloader))
    print("Batch image shape:", images.shape)
    for i in range(len(images)):
        img = images[i]
        lms = labels['landmarks'][i].cpu().numpy()
        print(labels['landmarks'][i].shape)
        
        emotion = labels['emotion'][i].item()
        show_image_with_landmarks(img, lms, title=f"Emotion: {emotion}")
        
        
        
        
        

class FaceDataset_2(Dataset):
    def __init__(self, df, image_size=128, training=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.training = training

        self.augment = A.Compose([
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(image_size, image_size)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.no_aug = A.Compose([
            A.Resize(image_size, image_size)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {row.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Crop around face ---
        x1, y1, x2, y2 = map(float, row.bbox)
        w, h = x2 - x1, y2 - y1

        if self.training:
            expand_ratio = np.random.uniform(0.05, 0.15)
        else:
            expand_ratio = 0.1

        x1 = max(0, x1 - w * expand_ratio)
        y1 = max(0, y1 - h * expand_ratio)
        x2 = min(image.shape[1], x2 + w * expand_ratio)
        y2 = min(image.shape[0], y2 + h * expand_ratio)

        H, W = image.shape[:2]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))
        image = image[y1:y2, x1:x2]

        landmarks = np.array(row['landmarks_manual'], dtype=np.float32)
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1

        h0, w0 = image.shape[:2]
        landmarks[:, 0] = np.clip(np.nan_to_num(landmarks[:, 0]), 0, w0 - 1)
        landmarks[:, 1] = np.clip(np.nan_to_num(landmarks[:, 1]), 0, h0 - 1)

        if self.training: 
            aug = self.augment(image=image, keypoints=landmarks.tolist())
        else: 
            aug = self.no_aug(image=image, keypoints=landmarks.tolist())

        image = aug['image']
        landmarks = np.array(aug['keypoints'], dtype=np.float32)

        h_crop, w_crop = image.shape[:2]
        landmarks[:, 0] /= w_crop
        landmarks[:, 1] /= h_crop

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        labels = {
            'emotion': torch.tensor(row.emotion, dtype=torch.long),
            'race': torch.tensor(row.race, dtype=torch.long),
            'gender': torch.tensor(row.gender, dtype=torch.long),
            'age': torch.tensor(row.age, dtype=torch.long),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32)
        }

        return image, labels
        
        
        
        
        

# class FaceDataset(Dataset):
#     def __init__(self, df, image_size=128, training=True):
#         self.df = df.reset_index(drop=True)
#         self.image_size = image_size
#         self.training = training

#         self.augment = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             # Use Affine with rotate to avoid Rotate duplicating keypoints
#             A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
#             A.RandomBrightnessContrast(p=0.3),
#             A.Resize(image_size, image_size)
#         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

#         self.no_aug = A.Compose([
#             A.Resize(image_size, image_size)
#         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

#         self.left_right_pairs = [(0, 1), (3, 4)]

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image = cv2.imread(row.image_path)
#         if image is None:
#             raise ValueError(f"Failed to read image: {row.image_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         x1, y1, x2, y2 = row.bbox
#         w, h = x2 - x1, y2 - y1
#         # print(f"Original bbox: {row.bbox}")
#         if self.training:
#             expand_ratio = np.random.uniform(0.05, 0.15)
#             x1 = max(0, x1 - int(w * expand_ratio))
#             y1 = max(0, y1 - int(h * expand_ratio))
#             x2 = min(image.shape[1], x2 + int(w * expand_ratio))
#             y2 = min(image.shape[0], y2 + int(h * expand_ratio))

#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         image = image[y1:y2, x1:x2]

#         landmarks = np.array(row['landmarks_manual'], dtype=np.float32)
#         landmarks[:, 0] -= x1
#         landmarks[:, 1] -= y1
        
#         # clip to crop bounds before Albumentations
#         h0, w0 = image.shape[:2]
#         landmarks[:, 0] = np.clip(landmarks[:, 0], 0, w0 - 1e-4)
#         landmarks[:, 1] = np.clip(landmarks[:, 1], 0, h0 - 1e-4)

#         if self.training:
#             aug = self.augment(image=image, keypoints=landmarks.tolist())
#             image = aug['image']
#             landmarks = np.array(aug['keypoints'], dtype=np.float32)

#         else:
            
#             aug = self.no_aug(image=image, keypoints=landmarks.tolist())
#             image = aug['image']
#             landmarks = np.array(aug['keypoints'], dtype=np.float32)


#         h_crop, w_crop = image.shape[:2]
#         landmarks[:, 0] /= w_crop
#         landmarks[:, 1] /= h_crop

#         image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

#         labels = {
#             'emotion': torch.tensor(row.emotion, dtype=torch.long),
#             'race': torch.tensor(row.race, dtype=torch.long),
#             'gender': torch.tensor(row.gender, dtype=torch.long),
#             'age': torch.tensor(row.age, dtype=torch.long),
#             'landmarks': torch.tensor(landmarks, dtype=torch.float32)
#         }

#         return image, labels


