import torch
import torch.nn as nn


def fgsm(model, loss_fn, images, labels, epsilon):
    images = images.clone().detach().to(next(model.parameters()).device)
    labels = labels.clone().detach().to(next(model.parameters()).device)
    images.requires_grad = True

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.detach().sign()
    perturbed_images = images + epsilon * grad_sign
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images
