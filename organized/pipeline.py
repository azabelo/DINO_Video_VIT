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
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

from random_tensor import yt_to_tensor
from organized.DINO_Video_Transforms import DINOVideoTransform
from vit_pytorch import ViT
from organized.get_dataset import get_hmdb51_dataset

class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        input_dim = 1024

        # Instantiate the Vision Transformer model
        model = ViT(
            image_size=224,  # Input image size
            image_time=80,  # Input image time
            patch_size=16,  # Patch size
            patch_time=8,  # Patch time
            num_classes=input_dim,  # Number of output classes
            dim=768,  # Embedding dimension
            depth=12,  # Number of transformer blocks
            heads=12,  # Number of attention heads
            mlp_dim=3072  # Hidden dimension of the MLP
        )
        backbone = model

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        print("forward student")
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        print("forward teacher")
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        print("training step")
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optim


def pretrain():
    print("starting pretraining")

    dataset = get_hmdb51_dataset()

    # dino_transform = DINOVideoTransform(global_crop_size=(224,224,160), local_crop_size=(224,224,160))
    dataset = LightlyDataset.from_torch_dataset(dataset)

    model = DINO()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=12,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)


    # we need to reactivate requires grad to perform supervised backpropagation later
    activate_requires_grad(model.teacher_backbone)
    return model.student_backbone

if __name__ == "__main__":
    pretrained_model = pretrain()