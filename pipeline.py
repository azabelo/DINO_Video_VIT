import copy
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torchvision
from torch import nn
import torchvision.transforms as transforms
from lightly.loss import DINOLoss
from lightly.data import LightlyDataset
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, activate_requires_grad
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vit_modified import ViT
from get_dataset import get_hmdb51_dataset
from DINO_Video_Transforms import DINOVideoTransform
from video_transforms import resize_video

class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        input_dim = 1024

        # Instantiate the Vision Transformer model
        backbone = ViT(
            image_size=112,  # Input image size
            image_time=40,  # Input image time
            patch_size=16,  # Patch size
            patch_time=8,  # Patch time
            num_classes=input_dim,  # Number of output classes
            dim=768,  # Embedding dimension
            depth=12,  # Number of transformer blocks
            heads=4,  # Number of attention heads
            mlp_dim=1024  # Hidden dimension of the MLP
        )

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 2048, 256, 2048
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, 2048, 256, 2048
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        wandb.log({"pretraining_loss": loss})
        gpu_memory = torch.cuda.memory_allocated()
        wandb.log({"GPU Memory": gpu_memory})
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def set_params(self, lr_factor, max_epochs):
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

    def configure_optimizers(self):
        param = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * self.lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [cosine_scheduler]


class Classifier(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.feature_extractor = model
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

class Supervised_trainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=51)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # ignore audio
        x, z, y = batch
        print(z)
        print(z.shape)
        print(x.shape)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #ignore audio
        x, _, y = batch
        print(y)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_acc', acc, prog_bar=False)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-5)
        return optimizer


def pretrain(path_to_hmdb51, args):
    print("starting pretraining")
    wandb.init(project='DINO Video Pretraining')

    dino_transform = DINOVideoTransform(global_crop_size=(112, 112, 40), local_crop_size=(112, 112, 40))

    dataset = get_hmdb51_dataset(path_to_hmdb51, dino_transform)
    dataset = LightlyDataset.from_torch_dataset(dataset)

    #params
    bs = args.batch_size
    num_workers = 32
    lr_factor = bs / 256
    max_epochs = args.pretrain_epochs

    model = DINO()
    model.set_params(lr_factor, max_epochs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

    wandb.finish()

    # we need to reactivate requires grad to perform supervised backpropagation later
    activate_requires_grad(model.student_backbone)
    return model.student_backbone

class Resize_Transform():
    def __init__(self, space_size, time_size):
        self.space_size = space_size
        self.time_size = time_size

    def __call__(self, video):
        return resize_video(video, self.space_size, self.time_size).permute(1,0,2,3) #permute to get CxTxHxW

def supervised_train(model, path_to_hmdb51, args):
    print("starting sup training")
    wandb.init(project='HMDB-51 video classification')
    # Log the arguments to wandb
    wandb.config.update(args)

    #create a transform to resize the video to 112x112x40
    resize_transform = Resize_Transform(40,112)
    dataset = get_hmdb51_dataset(path_to_hmdb51, resize_transform)
    dataset = LightlyDataset.from_torch_dataset(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=32)

    sup_trainer = Supervised_trainer(model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    wandb_logger = WandbLogger(project='sup training', log_model=True)
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=args.supervised_epochs, devices=1, accelerator=accelerator, logger=wandb_logger)

    # Train the model
    trainer.fit(sup_trainer, train_loader, val_dataloaders=val_loader)

    wandb.finish()
def show_video(tensor):
    """
    Show a video

    Args:
        tensor: video tensor of shape (T, C, H, W)
    """
    fig = plt.figure()
    ims = []
    for i in range(tensor.shape[0]):
        im = plt.imshow(tensor[i].permute(1, 2, 0))
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='hmdb51_unrared')
    parser.add_argument('--pretrain_epochs', type=int, default=1)
    parser.add_argument('--supervised_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sup_batch_size', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #prevent division by zero in video time?
    args = getArgs()
    pretrained_model = pretrain(args.path, args)
    classifier = Classifier(pretrained_model, 51)
    supervised_train(classifier, args.path, args)