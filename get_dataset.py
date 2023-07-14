import os
import glob
import torch
from torchvision.io import read_video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from DINO_Video_Transforms import DINOVideoTransform

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
        #video_frames, _, _ = read_video(video_path, pts_unit="sec")
        video_frames = self.video_to_tensor(video_path)

        # Convert video frames to tensor
        frames = []
        for frame in video_frames:
            # Convert frame to NumPy array
            frame = np.array(frame)
            # Resize frame if needed (done in transforms later so not needed here)

            # Convert frame to torch tensor
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float()

            frames.append(frame)

        video_tensor = torch.stack(frames)  # Stack frames along the time dimension
        # #apply transform to video (there should at least be resizing)
        video_tensor = self.transform(video_tensor)

        return video_tensor, label

    def video_to_tensor(self, file_path):
        # Open the video file
        video = cv2.VideoCapture(file_path)

        frames = []
        while True:
            # Read a frame from the video
            ret, frame = video.read()

            # Break the loop if no more frames are available
            if not ret:
                break

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PyTorch tensor
            frame_tensor = torch.from_numpy(frame_rgb)

            # Append the frame tensor to the list of frames
            frames.append(frame_tensor)

        # Stack the frames tensor along the time dimension
        video_tensor = torch.stack(frames, dim=0)

        return video_tensor

def get_hmdb51_dataset(path_to_hmdb51, dino_transform):

    #path_to_hmdb51 = "other path"
    dataset = CustomDataset(path_to_hmdb51, dino_transform)
    return dataset

#create a dataloader for the dataset
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

# Access the video and label of the first sample
# video, label = dataset[1]
# video = video.to(torch.int)
# print(label)
# print(video.shape)
# print(video[0])
#
# fig, ax = plt.subplots()
#
# imgs = video.permute(0,2,3,1) # Permuting to HxWxC format
# frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]
#
# ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
#
# plt.show()

