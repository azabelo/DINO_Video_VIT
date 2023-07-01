from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torchvision
import av
import os
import cv2

dataset_file = "/home/andrewzabelo/PycharmProjects/DINO_Video_VIT/hmdb51_dataset.pt"
if not os.path.exists(dataset_file) or True:
    dataset = torchvision.datasets.HMDB51(root = "/home/andrewzabelo/Documents/hmdb51_extracted",
                                      annotation_path= "/home/andrewzabelo/Documents/testTrainMulti_7030_splits",
                                     frames_per_clip=32, output_format="TCHW", num_workers=12)
    torch.save(dataset, dataset_file)
else:
    dataset = torch.load(dataset_file)

#print the size of the dataset and the number of classes
print("Dataset size: ", len(dataset))
video = dataset[0][0]
print(dataset.classes[dataset[0][2]])



fig, ax = plt.subplots()

imgs = video.permute(0,2,3,1) # Permuting to HxWxC format
frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=False)

plt.show()