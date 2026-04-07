import torch
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import json
import torch.nn.functional as F

# ---------------- LOAD CLASS LABELS ----------------
with open("class_labels.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {int(v): k for k, v in class_to_idx.items()}

# ---------------- LOAD MODEL ----------------
num_classes = len(class_to_idx)

model = mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

model.load_state_dict(torch.load("mobilenetv3_plant_disease.pth", map_location='cpu'))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return idx_to_class[pred.item()], confidence.item()