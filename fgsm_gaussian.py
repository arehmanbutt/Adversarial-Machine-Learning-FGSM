import torch
import torch.nn as nn

def fgsm_gaussian(images, epsilon):
    images = images.clone().detach()
    noise = torch.randn_like(images) * epsilon
    perturbed_images = images + noise
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images
