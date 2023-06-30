from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torchvision
import deeplake

ds = deeplake.load("hub://activeloop/hmdb51-train")
dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
print(type(dataloader))

print(type(dataloader.dataset[0]))


fig, ax = plt.subplots()

imgs = torch.rand(32,3,224,224)
imgs = imgs.permute(0,2,3,1) # Permuting to (Bx)HxWxC format
frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]

ani = animation.ArtistAnimation(fig, frames, interval=50)

plt.show()