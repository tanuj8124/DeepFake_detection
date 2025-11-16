#!/usr/bin/env python3
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import nn
from torchvision import models
import numpy as np
import cv2
import mediapipe as mp
import argparse
import json
import sys

# Mediapipe Face Detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --------------------------------------------------------
# Model definition (same as your original)
# --------------------------------------------------------
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# --------------------------------------------------------
# Mediapipe based cropper (replaces dlib)
# --------------------------------------------------------
def crop_face_mediapipe(frame):
    h, w, _ = frame.shape
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None

    det = results.detections[0]
    box = det.location_data.relative_bounding_box

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    # Safety
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return face

# --------------------------------------------------------
# Dataset class
# --------------------------------------------------------
class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frame_extract(self.video_path)):

            face = crop_face_mediapipe(frame)
            if face is not None:
                frame = face

            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        
        # pad frames
        while len(frames) < self.count:
            frames.append(frames[-1] if frames else torch.zeros(3, 112, 112))
        
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()

# --------------------------------------------------------
# Prediction function
# --------------------------------------------------------
def predict(model, video_tensor, device='cuda'):
    model.eval()
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        _, logits = model(video_tensor)
        probs = nn.Softmax(dim=1)(logits)

        _, prediction = torch.max(probs, 1)
        confidence = probs[0, int(prediction.item())].item() * 100

        return {
            'prediction': int(prediction.item()),
            'confidence': round(confidence, 2)
        }

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Prediction')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--sequence-length', type=int, default=20, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        device = args.device if torch.cuda.is_available() else 'cpu'

        # transforms
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # load model
        model = Model(2).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        # dataset
        dataset = ValidationDataset(
            args.video,
            sequence_length=args.sequence_length,
            transform=transform
        )
        
        video_tensor = dataset[0]
        result = predict(model, video_tensor, device)

        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'prediction': -1,
            'confidence': 0.0
        }), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
