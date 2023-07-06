import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, ToPILImage, ToTensor

def random_time_crop(video, min_crop_ratio, max_crop_ratio):
    """
    Randomly crop a video in the time dimension

    Args:
        video: video tensor of shape (T, C, H, W)
        min_crop_ratio: minimum crop ratio in the time dimension
        max_crop_ratio: maximum crop ratio in the time dimension

    Returns:
        video tensor of shape (T, C, H, W)
    """
    time_length = video.shape[0]
    assert time_length >= 2
    crop_ratio = torch.rand((1,)).item() * (max_crop_ratio - min_crop_ratio) + min_crop_ratio
    crop_length = int(time_length * crop_ratio)
    start_idx = torch.randint(low=0, high=time_length - crop_length + 1, size=(1,)).item()
    return video[start_idx:start_idx + crop_length, :, :, :]

def random_space_crop(video, min_crop_ratio, max_crop_ratio):
    """
    Randomly crop a video in the space dimension

    Args:
        video: video tensor of shape (T, C, H, W)
        min_crop_ratio: minimum crop ratio in the space dimension
        max_crop_ratio: maximum crop ratio in the space dimension

    Returns:
        video tensor of shape (T, C, H, W)
    """
    height = video.shape[2]
    width = video.shape[3]
    assert height >= 2 and width >= 2
    crop_ratio = torch.rand((1,)).item() * (max_crop_ratio - min_crop_ratio) + min_crop_ratio
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    start_height_idx = torch.randint(low=0, high=height - crop_height + 1, size=(1,)).item()
    start_width_idx = torch.randint(low=0, high=width - crop_width + 1, size=(1,)).item()
    return video[:, :, start_height_idx:start_height_idx + crop_height, start_width_idx:start_width_idx + crop_width]

def resize_video(video_tensor, time_size, spatial_size):
    # Get the current time and spatial dimensions
    original_time_size = video_tensor.shape[0]
    original_spatial_size = video_tensor.shape[2:]

    # Resize the video in the time dimension
    if original_time_size != time_size:
        if original_time_size < time_size:
            # Upsample the video in the time dimension
            video_tensor = F.interpolate(video_tensor.unsqueeze(0), size=(time_size,) + original_spatial_size, mode='nearest').squeeze(0)
        else:
            # Downsample the video in the time dimension
            indices = torch.linspace(0, original_time_size - 1, time_size).long()
            video_tensor = video_tensor[indices]

    # Resize the video in the spatial dimensions
    if original_spatial_size != spatial_size:
        video_tensor = F.interpolate(video_tensor, size=spatial_size, mode='bilinear', align_corners=False)

    return video_tensor

def random_horizontal_flip(video, p):
    """
    #flip every frae of the video one at a time
    """
    if torch.rand((1,)).item() < p:
        return torch.flip(video, dims=(3,))
    return video


def random_vertical_flip(video, p):
    """
    Randomly flip a video vertically

    Args:
        video: video tensor of shape (T, C, H, W)
        p: probability of flipping

    Returns:
        video tensor of shape (T, C, H, W)
    """
    if torch.rand((1,)).item() < p:
        return torch.flip(video, dims=(3,))
    return video



def random_rotation(video, p, min_degrees, max_degrees):
    """
    Randomly rotate a video

    Args:
        video: video tensor of shape (T, C, H, W)
        p: probability of rotating
        min_degrees: minimum rotation degrees
        max_degrees: maximum rotation degrees

    Returns:
        video tensor of shape (T, C, H, W)
    """
    if torch.rand((1,)).item() < p:
        degrees = torch.rand((1,)).item() * (max_degrees - min_degrees) + min_degrees
        # Apply the transformation to each frame in the video tensor
        rotated_frames = []
        for frame in video:
            rotated_frame = TF.rotate(frame, degrees)
            rotated_frames.append(rotated_frame)
        # Convert the list of frames back to a video tensor
        rotated_video = torch.stack(rotated_frames)
        return rotated_video
    return video

def random_jitter(video_tensor, probability, brightness, contrast, saturation, hue):
    if torch.rand(1) < probability:
        # Initialize the ColorJitter transformation
        color_jitter = ColorJitter(brightness=(brightness,brightness),
                                   contrast=(contrast, contrast),
                                   saturation=(saturation,saturation),
                                   hue=(hue,hue))

        # Initialize the ToPILImage and ToTensor transformations
        to_pil = ToPILImage()
        to_tensor = ToTensor()

        # Apply random color jittering to each frame in the video tensor
        jittered_frames = []

        for frame in video_tensor:
            # Convert the frame tensor to a PIL image
            pil_image = to_pil(frame)

            # Apply color jittering with the specified probability
            pil_image = color_jitter(pil_image)

            # Convert the modified PIL image back to a tensor
            jittered_frame = to_tensor(pil_image)

            # Append the jittered frame to the list
            jittered_frames.append(jittered_frame)

        # Stack the jittered frames along the time dimension
        jittered_video_tensor = torch.stack(jittered_frames)
        return jittered_video_tensor

    return video_tensor

def random_grayscale(video_tensor, p):
    if torch.rand((1,)).item() < p:
        # Initialize the ToPILImage and ToTensor transformations
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        # Apply the grayscale transformation to each frame in the video tensor
        grayscale_frames = []
        for frame in video_tensor:
            pil_image = to_pil(frame)
            pil_frame = TF.to_grayscale(pil_image,3)
            grayscale_frame = to_tensor(pil_frame)
            grayscale_frames.append(grayscale_frame)
        # Convert the list of frames back to a video tensor
        grayscale_video = torch.stack(grayscale_frames)
        return grayscale_video
    return video_tensor

def random_solarize(video, p, threshold):
    if torch.rand((1,)).item() < p:
        # Apply the solarize transformation to each frame in the video tensor
        solarized_frames = []
        for frame in video:
            solarized_frame = TF.solarize(frame, threshold)
            solarized_frames.append(solarized_frame)
        # Convert the list of frames back to a video tensor
        solarized_video = torch.stack(solarized_frames)
        return solarized_video
    return video

def random_gaussian_blur(video, p, sigma):
    if torch.rand((1,)).item() < p:
        # Apply the gaussian blur transformation to each frame in the video tensor
        blurred_frames = []
        sig = torch.rand((1,)).item() * (sigma[1] - sigma[0]) + sigma[0]
        for frame in video:
            blurred_frame = TF.gaussian_blur(frame, kernel_size=3, sigma=sig)
            blurred_frames.append(blurred_frame)
        # Convert the list of frames back to a video tensor
        blurred_video = torch.stack(blurred_frames)
        return blurred_video
    return video

def normalize(video, mean, std):
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    """
    Normalize a video

    Args:
        video: video tensor of shape (T, C, H, W)

    Returns:
        video tensor of shape (T, C, H, W)
    """
    video = video.float()
    normalized_frames = []
    for frame in video:
        normalized_frame = TF.normalize(frame, mean, std)
        normalized_frames.append(normalized_frame)
    normalized_video = torch.stack(normalized_frames)
    return normalized_video