from typing import Optional, Tuple, Union

import PIL
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE

import video_transforms as video_transforms


class DINOVideoTransform(MultiViewTransform):
    """Implements the global and local view augmentations for DINO [0].

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        global_crop_size: Tuple[int, int, int] = (224,224,160),
        global_crop_scale_space: Tuple[float, float] = (0.4, 1.0),
        global_crop_scale_time: Tuple[float, float] = (0.4, 1.0),
        local_crop_size: Tuple[int, int, int] = (224,224,160),
        local_crop_scale_space: Tuple[float, float] = (0.05, 0.4),
        local_crop_scale_time: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        # first global crop
        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale_space=global_crop_scale_space,
            crop_scale_time=global_crop_scale_time,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[0],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )

        # second global crop
        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale_space=global_crop_scale_space,
            crop_scale_time=global_crop_scale_time,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[1],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=solarization_prob,
            normalize=normalize,
        )

        # transformation for the local small crops
        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale_space=local_crop_scale_space,
            crop_scale_time=local_crop_scale_time,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[2],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )
        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


class DINOViewTransform:
    def __init__(
        self,
        crop_size: Tuple[int,int,int] = (224,224,160),
        crop_scale_space: Tuple[float, float] = (0.4, 1.0),
        crop_scale_time: Tuple[float, float] = (0.4, 1.0),
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        #initialize all these params to self
        self.crop_size = crop_size
        self.crop_scale_space = crop_scale_space
        self.crop_scale_time = crop_scale_time
        self.hf_prob = hf_prob
        self.vf_prob = vf_prob
        self.rr_prob = rr_prob
        self.rr_degrees = rr_degrees
        self.cj_prob = cj_prob
        self.cj_strength = cj_strength
        self.cj_bright = cj_bright
        self.cj_contrast = cj_contrast
        self.cj_sat = cj_sat
        self.cj_hue = cj_hue
        self.random_gray_scale = random_gray_scale
        self.gaussian_blur = gaussian_blur
        self.kernel_size = kernel_size
        self.kernel_scale = kernel_scale
        self.sigmas = sigmas
        self.solarization_prob = solarization_prob
        self.normalize = normalize


    def __call__(self, video: Tensor) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """

        video = video_transforms.random_space_crop(video,self.crop_scale_space[0], self.crop_scale_space[1])
        video = video_transforms.random_time_crop(video,self.crop_scale_time[0], self.crop_scale_time[1])
        video = video_transforms.resize_video(video, self.crop_size[2], self.crop_size[1])
        video = video_transforms.random_horizontal_flip(video, self.hf_prob)
        video = video_transforms.random_vertical_flip(video, self.vf_prob)
        video = video_transforms.random_rotation(video, self.rr_prob, 0, self.rr_degrees)
        video = video_transforms.random_jitter(video, self.cj_prob, self.cj_strength * self.cj_bright,
                                               self.cj_strength * self.cj_contrast,
                                               self.cj_strength * self.cj_sat,
                                               self.cj_strength * self.cj_hue)
        video = video_transforms.random_grayscale(video, self.random_gray_scale)
        video = video_transforms.random_gaussian_blur(video, self.gaussian_blur, self.sigmas)
        video = video_transforms.random_solarize(video, self.solarization_prob)
        # maybe include normalize
        video = video.permute(1,0,2,3)
        return video
