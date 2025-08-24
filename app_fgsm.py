from fastapi import FastAPI, UploadFile, Form
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fgsm import fgsm
from fgsm_gaussian import fgsm_gaussian
from model import SimpleCNN

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


@app.post("/fgsm_attack/")
async def fgsm_attack(
    file: UploadFile, label: int = Form(...), epsilon: float = Form(...)
):
    try:
        image = Image.open(file.file)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
        loss_fn = nn.CrossEntropyLoss()
        perturbed = fgsm(model, loss_fn, image, label_tensor, epsilon)

        with torch.no_grad():
            output = model(perturbed)
            pred = output.argmax(dim=1, keepdim=True).item()

        success = pred != label

        return {
            "Original_label": label,
            "Predicted_label": pred,
            "Attack success": success,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/fgsm_gaussian/")
async def fgsm_gaussian_attack(
    file: UploadFile, label: int = Form(...), epsilon: float = Form(...)
):
    try:
        image = Image.open(file.file)
        image = transform(image).unsqueeze(0).to(device)
        perturbed = fgsm_gaussian(image, epsilon)

        with torch.no_grad():
            output = model(perturbed)
            pred = output.argmax(dim=1, keepdim=True).item()

        success = pred != label

        return {
            "Original_label": label,
            "Predicted_label": pred,
            "Attack success": success,
        }

    except Exception as e:
        return {"error": str(e)}
