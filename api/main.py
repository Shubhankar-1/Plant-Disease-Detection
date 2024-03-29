import tensorflow as tf
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image

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

model = tf.keras.models.load_model("./3")

plant_disease_info = {
    0: {
        "plant_name": "Apple",
        "disease_name": "Apple Scab",
        "detailed_description": "Apple scab is a common disease of apple trees, caused by the fungus Venturia inaequalis. It typically appears as olive-green to black lesions on leaves and fruit, often with a velvety texture. Severe infections can lead to defoliation and reduced fruit quality.",
        "cure": "Prune infected branches to improve air circulation, apply fungicides during the growing season, and practice good sanitation by removing fallen leaves and fruit to reduce overwintering sources of the fungus.",
    },
    1: {
        "plant_name": "Apple",
        "disease_name": "Black Rot",
        "detailed_description": "Black rot is a fungal disease caused by the pathogen Botryosphaeria obtusa. It affects apple trees during warm, wet weather, causing dark, sunken lesions on fruit and leaves, as well as cankers on branches.",
        "cure": "Prune infected branches, remove mummified fruit, apply fungicides preventively, and practice good orchard sanitation to reduce disease spread.",
    },
    2: {
        "plant_name": "Apple",
        "disease_name": "Cedar Apple Rust",
        "detailed_description": "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae. It affects both apples and junipers, with symptoms including yellow-orange spots on leaves, as well as cankers and distortions on fruit.",
        "cure": "Prune infected branches, remove nearby junipers if possible, apply fungicides preventively, and promote good air circulation to reduce moisture levels.",
    },
    3: {
        "plant_name": "Apple",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    4: {
        "plant_name": "Blueberry",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    5: {
        "plant_name": "Cherry (including sour)",
        "disease_name": "Powdery Mildew",
        "detailed_description": "Powdery mildew is a fungal disease caused by various species of fungi in the Podosphaera and Erysiphe genera. It appears as white powdery patches on leaves, stems, and fruit, leading to stunted growth and reduced yield.",
        "cure": "Apply fungicides preventively, prune infected plant parts, and promote good air circulation to reduce humidity levels.",
    },
    6: {
        "plant_name": "Cherry (including sour)",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    7: {
        "plant_name": "Corn (maize)",
        "disease_name": "Cercospora Leaf Spot and Gray Leaf Spot",
        "detailed_description": "Cercospora leaf spot and gray leaf spot are fungal diseases caused by the pathogens Cercospora zeae-maydis and Cercospora zeina, respectively. They appear as small, tan to gray lesions with dark borders on leaves, leading to premature leaf death and reduced yield.",
        "cure": "Plant resistant varieties, apply fungicides preventively, practice crop rotation, and maintain proper plant spacing to improve air circulation.",
    },
    8: {
        "plant_name": "Corn (maize)",
        "disease_name": "Common Rust",
        "detailed_description": "Common rust is a fungal disease caused by the pathogen Puccinia sorghi. It appears as small, circular to elongated orange pustules on leaves, which can coalesce and cause leaf yellowing and premature death.",
        "cure": "Plant resistant varieties, apply fungicides preventively, and practice crop rotation to reduce disease pressure.",
    },
    9: {
        "plant_name": "Corn (maize)",
        "disease_name": "Northern Leaf Blight",
        "detailed_description": "Northern leaf blight is a fungal disease caused by the pathogen Exserohilum turcicum. It appears as cigar-shaped lesions with wavy margins on leaves, leading to premature leaf death and reduced yield.",
        "cure": "Plant resistant varieties, apply fungicides preventively, and practice crop rotation to reduce disease pressure.",
    },
    10: {
        "plant_name": "Corn (maize)",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    11: {
        "plant_name": "Grape",
        "disease_name": "Black Rot",
        "detailed_description": "Black rot is a fungal disease caused by the pathogen Guignardia bidwellii. It affects grapevines during warm, wet weather, causing dark, sunken lesions on fruit and leaves, which can lead to defoliation and reduced yield.",
        "cure": "Prune infected vines, remove mummified fruit, apply fungicides preventively, and practice good vineyard sanitation to reduce disease spread.",
    },
    12: {
        "plant_name": "Grape",
        "disease_name": "Esca (Black Measles)",
        "detailed_description": "Esca, also known as black measles, is a fungal disease complex caused by several pathogens, including Phaeomoniella chlamydospora and Phaeoacremonium spp. It leads to yellowing and necrosis of leaves, as well as wood decay in grapevines.",
        "cure": "Prune infected vines, apply fungicides preventively, and practice good vineyard sanitation to reduce disease spread.",
    },
    13: {
        "plant_name": "Grape",
        "disease_name": "Leaf Blight (Isariopsis Leaf Spot)",
        "detailed_description": "Leaf blight, caused by the fungus Isariopsis spp., appears as circular to irregular brown lesions on grape leaves, which can lead to defoliation and reduced yield.",
        "cure": "Apply fungicides preventively, prune infected vines, and promote good air circulation to reduce humidity levels.",
    },
    14: {
        "plant_name": "Grape",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    15: {
        "plant_name": "Orange",
        "disease_name": "Haunglongbing (Citrus Greening)",
        "detailed_description": "Huanglongbing, also known as citrus greening, is a bacterial disease caused by the pathogen Candidatus Liberibacter asiaticus. It affects citrus trees, causing yellowing of leaves, stunted growth, and bitter, misshapen fruit.",
        "cure": "There is currently no cure for citrus greening. Management strategies include removing infected trees, controlling vector insects, and planting disease-free nursery stock.",
    },
    16: {
        "plant_name": "Peach",
        "disease_name": "Bacterial Spot",
        "detailed_description": "Bacterial spot is a bacterial disease caused by Xanthomonas arboricola pv. pruni. It appears as water-soaked lesions on leaves and fruit, which later turn dark and necrotic. Severe infections can lead to defoliation and reduced fruit quality.",
        "cure": "Apply copper-based fungicides preventively, prune infected branches, and practice good orchard sanitation to reduce disease spread.",
    },
    17: {
        "plant_name": "Peach",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    18: {
        "plant_name": "Pepper, bell",
        "disease_name": "Bacterial Spot",
        "detailed_description": "Bacterial spot is a bacterial disease caused by Xanthomonas spp. It appears as water-soaked lesions on leaves and fruit, which later turn dark and necrotic. Severe infections can lead to defoliation and reduced fruit quality.",
        "cure": "Apply copper-based fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    19: {
        "plant_name": "Pepper, bell",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    20: {
        "plant_name": "Potato",
        "disease_name": "Early Blight",
        "detailed_description": "Early blight is a fungal disease caused by the pathogen Alternaria solani. It appears as dark lesions with concentric rings on leaves, which can expand and cause defoliation. Tubers may also develop shallow, sunken lesions.",
        "cure": "Apply fungicides preventively, practice crop rotation, and promote good air circulation to reduce humidity levels.",
    },
    21: {
        "plant_name": "Potato",
        "disease_name": "Late Blight",
        "detailed_description": "Late blight is a fungal disease caused by Phytophthora infestans. It appears as water-soaked lesions on leaves, which rapidly turn brown and necrotic. Tubers may develop dark, firm lesions that can lead to rotting.",
        "cure": "Apply fungicides preventively, practice crop rotation, and promote good air circulation to reduce humidity levels.",
    },
    22: {
        "plant_name": "Potato",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    23: {
        "plant_name": "Raspberry",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    24: {
        "plant_name": "Soybean",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    25: {
        "plant_name": "Squash",
        "disease_name": "Powdery Mildew",
        "detailed_description": "Powdery mildew is a fungal disease caused by various species of fungi in the Podosphaera and Erysiphe genera. It appears as white powdery patches on leaves, stems, and fruit, leading to stunted growth and reduced yield.",
        "cure": "Apply fungicides preventively, prune infected plant parts, and promote good air circulation to reduce humidity levels.",
    },
    26: {
        "plant_name": "Strawberry",
        "disease_name": "Leaf Scorch",
        "detailed_description": "Leaf scorch is a physiological disorder caused by various factors, including environmental stress, nutrient deficiencies, and water imbalance. It appears as browning and necrosis of leaf margins, often starting at the tips.",
        "cure": "Address underlying causes such as watering and nutrient management, provide adequate mulching, and protect plants from extreme weather conditions.",
    },
    27: {
        "plant_name": "Strawberry",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
    28: {
        "plant_name": "Tomato",
        "disease_name": "Bacterial Spot",
        "detailed_description": "Bacterial spot is a bacterial disease caused by Xanthomonas spp. It appears as water-soaked lesions on leaves and fruit, which later turn dark and necrotic. Severe infections can lead to defoliation and reduced fruit quality.",
        "cure": "Apply copper-based fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    29: {
        "plant_name": "Tomato",
        "disease_name": "Early Blight",
        "detailed_description": "Early blight is a fungal disease caused by the pathogen Alternaria solani. It appears as dark lesions with concentric rings on leaves, which can expand and cause defoliation. Fruit may also develop lesions.",
        "cure": "Apply fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    30: {
        "plant_name": "Tomato",
        "disease_name": "Late Blight",
        "detailed_description": "Late blight is a fungal disease caused by Phytophthora infestans. It appears as water-soaked lesions on leaves, which rapidly turn brown and necrotic. Fruit may also develop lesions, and the entire plant can collapse.",
        "cure": "Apply fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    31: {
        "plant_name": "Tomato",
        "disease_name": "Leaf Mold",
        "detailed_description": "Leaf mold is a fungal disease caused by the pathogen Fulvia fulva. It appears as yellow to brown lesions on leaves, often with a fuzzy texture on the underside. Severe infections can lead to defoliation and reduced fruit quality.",
        "cure": "Promote good air circulation, avoid overhead watering, and apply fungicides preventively to reduce disease spread.",
    },
    32: {
        "plant_name": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "detailed_description": "Septoria leaf spot is a fungal disease caused by Septoria lycopersici. It appears as small, dark lesions with gray centers and yellow halos on tomato leaves, which can lead to defoliation and reduced yield.",
        "cure": "Apply fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    33: {
        "plant_name": "Tomato",
        "disease_name": "Spider Mites Two-Spotted Spider Mite",
        "detailed_description": "Two-spotted spider mites are common pests of tomatoes, feeding on plant sap and causing stippling, webbing, and eventual leaf yellowing and defoliation.",
        "cure": "Use insecticidal soaps or oils, introduce predatory mites or insects, and maintain proper plant hygiene to reduce mite populations.",
    },
    34: {
        "plant_name": "Tomato",
        "disease_name": "Target Spot",
        "detailed_description": "Target spot, also known as Corynespora leaf spot, is a fungal disease caused by Corynespora cassiicola. It appears as circular lesions with dark margins and light centers on tomato leaves, which can coalesce and lead to defoliation.",
        "cure": "Apply fungicides preventively, prune infected branches, and practice good garden sanitation to reduce disease spread.",
    },
    35: {
        "plant_name": "Tomato",
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "detailed_description": "Tomato yellow leaf curl virus (TYLCV) is a viral disease transmitted by whiteflies. It affects tomato plants, causing yellowing, curling, and stunting of leaves, as well as reduced fruit yield and quality.",
        "cure": "There is no cure for TYLCV. Control whitefly populations using insecticides, remove infected plants, and use resistant tomato varieties where available.",
    },
    36: {
        "plant_name": "Tomato",
        "disease_name": "Tomato Mosaic Virus",
        "detailed_description": "Tomato mosaic virus (ToMV) is a viral disease that affects tomatoes, causing mosaic patterns of light and dark green on leaves, as well as leaf distortion, stunting, and reduced fruit yield.",
        "cure": "There is no cure for ToMV. Control aphid populations using insecticides, remove infected plants, and use resistant tomato varieties where available.",
    },
    37: {
        "plant_name": "Tomato",
        "disease_name": "Healthy",
        "detailed_description": "The plant is healthy without any disease symptoms.",
        "cure": "No treatment needed, maintain good cultural practices to support plant health.",
    },
}


@app.get("/ping")
async def ping():
    return "Hello, Server is running"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = np.array(image.resize((256, 256)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = read_file_as_image(await file.read())

        # Prepare image for prediction
        image_batch = np.expand_dims(image, 0)

        # Make prediction
        predictions = model.predict(image_batch)

        # Get predicted class and confidence
        predicted_class = plant_disease_info[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        confidence_percentage = str(int(round(confidence * 100))) + "%"
        return {"data": predicted_class, "confidence": confidence_percentage}

    except (IOError, ValueError) as e:  # Catch specific exceptions
        return Response(
            content={"error": f"Error processing image: {str(e)}"}, status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
