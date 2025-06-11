import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
lesion_classes = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}

# Load EfficientNet-B3 model
def load_model():
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(lesion_classes))
    )
    model.load_state_dict(torch.load("best_efficientnet_b3.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Image preprocessing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict image class
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).item()

    return lesion_classes[predicted_class], probabilities

# Load model once
model = load_model()
