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
            video_files = glob.glob(os.path.join(cls_dir, "*.avi"))
            for video_file in video_files:
                samples.append((video_file, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # Read video frames
        video_frames, _, _ = read_video(video_path, pts_unit="sec")

        # Convert video frames to tensor
        frames = []
        for frame in video_frames:
            # Convert frame to NumPy array
            frame = np.array(frame)

            # Resize frame if needed
            frame = cv2.resize(frame, (224, 224))

            # Convert frame to torch tensor
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float()

            frames.append(frame)

        video_tensor = torch.stack(frames)  # Stack frames along the time dimension

        return video_tensor, label

# Example usage:
root_dir = "/home/andrewzabelo/Documents/hmdb51_extracted"

dataset = CustomDataset(root_dir)

# Access the video and label of the first sample
video, label = dataset[1]
video = video.to(torch.int)
print(label)
print(video.shape)
print(video[0])

fig, ax = plt.subplots()

imgs = video.permute(0,2,3,1) # Permuting to HxWxC format
frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]

ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False)

plt.show()
