import tensorflow as tf
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model("./4.h5")

# Load plant disease info
with open("plant_info.json", "r") as file:
    plant_disease_info = json.load(file)


@app.get("/")
async def ping():
    return "Hello, Server is running"


# Function to read the image file as numpy array
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = np.array(image.resize((256, 256)))
    return image


# Function to make prediction on image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = read_file_as_image(await file.read())

        # Prepare image for prediction
        image_batch = np.expand_dims(image, 0)

        # Make prediction
        predictions = model.predict(image_batch) if model else "Model not found"

        # Get predicted class and confidence
        predicted_class = plant_disease_info[str(np.argmax(predictions[0]))]
        confidence = float(np.max(predictions[0]))

        confidence_percentage = str(int(round(confidence * 100))) + "%"
        return {"data": predicted_class, "confidence": confidence_percentage}

    except (IOError, ValueError) as e:  # Catch specific exceptions
        return Response(
            content={"error": f"Error processing image: {str(e)}"}, status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
