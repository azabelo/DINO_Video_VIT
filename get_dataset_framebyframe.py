import os
import glob
import torch
from torchvision.io import read_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            video_dirs = sorted(glob.glob(os.path.join(cls_dir, '*')))  # Assuming each video is a subdirectory
            for video_dir in video_dirs:
                samples.append((video_dir, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        video_frames = []
        frame_files = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))  # Assuming frames are in JPEG format
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            video_frames.append(frame)

        # Convert video frames to tensor
        frames = []
        for frame in video_frames:
            # Resize frame if needed
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float()  # Transpose dimensions and convert to float tensor
            frames.append(frame)

        video_tensor = torch.stack(frames)  # Stack frames along the time dimension

        return video_tensor, label


# Example usage:
root_dir = "/home/andrewzabelo/Documents/HMDB51"

dataset = CustomDataset(root_dir)

# Access the video and label of the first sample
video, label = dataset[1]
video = video.to(torch.int)
print(label)
print(video.shape)

fig, ax = plt.subplots()

imgs = video.permute(0,2,3,1) # Permuting to HxWxC format
frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]

ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False)

plt.show()