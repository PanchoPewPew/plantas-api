from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import uuid

import torch
import timm
from torchvision import transforms
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_ANON_KEY"]
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classes = ["disease", "healthy"]

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "plant_disease_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"status": "API funcionando"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[
        prediction.item()
    ]

    confidence_pct = round(
        confidence.item() * 100,
        2
    )

    saved = False
    inspection_id = None

    try:
        file_ext = file.filename.split(".")[-1] if file.filename else "jpg"
        image_path = f"inspections/{uuid.uuid4()}.{file_ext}"

        supabase.storage.from_("plant-images").upload(
            image_path,
            image_bytes,
            {"content-type": f"image/{file_ext}"}
        )

        result = supabase.table("inspections").insert({
            "image_path": image_path,
            "prediction_class": predicted_class,
            "confidence": confidence_pct
        }).execute()

        inspection_id = result.data[0]["id"]
        saved = True
    except Exception:
        pass

    return {
        "class": predicted_class,
        "confidence": confidence_pct,
        "saved": saved,
        "inspection_id": inspection_id
    }