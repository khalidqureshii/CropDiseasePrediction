from IPython.display import Markdown, display # type: ignore
from google import genai
from openai import OpenAI # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import re

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# --- Options ---
crop_options = "Wheat, Mustard, Potato"
disease_options = """Healthy, Aphid, Black Rust, Brown Rust, Blast Test, Leaf Blight,
Common Root Rot, Fusarium Head Blight, Mildew, Mite, Septoria, Smut, Stem Fly, Tan Spot, Yellow Rust, None of the Above"""

# --- Step 1: Detailed image description ---
def describe_image_with_gemini(img_bytes):
    prompt = """
    You are a highly experienced agricultural expert and botanist. Carefully observe the uploaded image of a crop leaf or plant.
    Provide a detailed textual description suitable for another AI model to understand the plant and predict its crop type and disease.
    Include leaf shape, color, spots, lesions, abnormalities, stem and flower details, and any visible symptoms.
    Avoid guessing the crop or disease; focus on direct observation, ignore watermarks or any text at corners.
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_bytes}}]
        }]
    )
    return response.text.strip()

# --- Step 2: Identify crop and disease (Flash) ---
def identify_crop_and_disease(img_bytes, mime_type):
    prompt = f"""
        You are an agricultural expert. Identify the crop and disease in this image. Ignore watermarks or any text at corners.

        Crops: {crop_options}
        Diseases: {disease_options}

        Respond in the format: Crop: <name>, Disease: <name>. I repeat, I do not want any sentences, just answer in the format: Crop: <name>, Disease: <name>.
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]
        }]
    )
    return response.text.strip()

# --- Step 3: Identify crop and disease (Pro) ---
def pro_identify_crop_and_disease(img_bytes, mime_type):
    prompt = f"""
        You are an agricultural expert. Identify the crop and disease in this image. Ignore watermarks or any text at corners.

        Crops: {crop_options}
        Diseases: {disease_options}

        Respond in the format: Crop: <name>, Disease: <name>. I repeat, I do not want any sentences, just answer in the format: Crop: <name>, Disease: <name>.
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]
        }]
    )
    return response.text.strip()

# --- Step 4: Text-only models ---
def predict_with_text_models(description, model_list):
    predictions = {}
    prompt = f"""
    Based on this description:
    {description}

    Identify the crop (options: {crop_options}).
    Identify the disease (options: {disease_options}).

    Respond in the format: Crop: <name>, Disease: <name>. I repeat, I do not want any sentences, just answer in the format: Crop: <name>, Disease: <name>.
    """
    for model in model_list:
        try:
            response = openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an agricultural expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            predictions[model] = response.choices[0].message.content.strip()
        except Exception as e:
            predictions[model] = f"Error: {str(e)}"
    return predictions

# --- Step 5: Verify Output ---
def verify_output(img_bytes, conflicting_opinions, mime_type):
    opinions_text = "\n".join([f"- {op}" for op in conflicting_opinions]) if conflicting_opinions else "None"
    prompt = f"""
    You are a highly reliable agricultural expert. Carefully analyze the uploaded image to identify the crop and disease.

    Other AI models provided these conflicting predictions:
    {opinions_text}

    Use the image as the main source of truth. The conflicting opinions are hints only.
    If none of the predictions fit what you see in the image, override them with your own judgment.

    Crops: {crop_options}
    Diseases: {disease_options}

    Respond in the format: Crop: <name>, Disease: <name>.
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]
        }]
    )
    return response.text.strip()

# Step 6: Suggest Causes and Recommendations
def get_disease_analysis(prediction: str):
    prompt = f"""
    You are an agricultural expert. The detected condition is:

    {prediction}

    Provide two sections:

    ## Possible Causes
    - List concise bullet points explaining why this disease might occur.
    - Keep each point within 2 lines.
    - Use numbers or specifics where possible (e.g., "High humidity >70%", "3-4 days of continuous rain").

    ## Recommendations
    - Give **only current, actionable steps** (no "if disease continues", no "in future seasons").
    - Specify **the best suited product(s)** (not "e.g.").
    - Include **dosage, frequency, and application method** (e.g., "Apply 2 sprays of Propiconazole 25 EC @ 1 ml/L water, 10 days apart").
    - Include any **field management practices** (like irrigation adjustments, removal of infected plants) that can be done immediately.
    - Keep each bullet ≤2 lines.

    Remember, give 2 clear sections 'Possible Causes' and 'Recommendations' with bullet points.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": prompt}]}]
    )

    text = response.text.strip()

    causes, recommendations = [], []
    current_section = None

    for line in text.splitlines():
        line = line.strip()

        # Section headers (flexible match)
        if re.search(r"possible causes", line, re.IGNORECASE):
            current_section = "causes"
            continue
        elif re.search(r"recommendations", line, re.IGNORECASE):
            current_section = "recommendations"
            continue

        # Bullets can start with -, *, or numbers like 1.
        if re.match(r"^[-*]\s+|^\d+\.\s+", line):
            item = re.sub(r"^[-*\d.]+\s*", "", line).strip()
            item = item.replace("**", "")
            if current_section == "causes":
                causes.append(item)
            elif current_section == "recommendations":
                recommendations.append(item)

    # Extract crop & disease
    try:
        parsed = prediction.replace("Crop:", "").replace("Disease:", "").split(",")
        crop = parsed[0].strip()
        disease = parsed[1].strip()
    except Exception:
        crop, disease = None, prediction.strip()

    return {
        "crop": crop,
        "disease": disease,
        "causes": causes,
        "recommendations": recommendations
    }


# --- Main Workflow ---
def analyze_disease(img_bytes, mime_type):
    description = describe_image_with_gemini(img_bytes)
    print(f"\n=== Gemini Description ===\n{description}")

    gemini_prediction = identify_crop_and_disease(img_bytes, mime_type)
    print(f"\n=== Gemini Flash Prediction ===\n{gemini_prediction}")

    # gemini_pro_prediction = pro_identify_crop_and_disease(img_bytes, mime_type)
    # print(f"\n=== Gemini Pro Prediction ===\n{gemini_pro_prediction}")

    # models = ["qwen/qwen3-coder:free", "deepseek/deepseek-chat-v3.1:free"]
    # predictions = predict_with_text_models(description, models)

    # conflicting_opinions = [gemini_prediction]
    # if gemini_prediction != gemini_pro_prediction:
    #     conflicting_opinions.append(gemini_pro_prediction)
    # for model, pred in predictions.items():
    #     if pred != gemini_prediction and len(pred) <= 50:
    #         conflicting_opinions.append(pred)

    # for model, pred in predictions.items():
    #     print(f"\n=== Model: {model} ===\nPrediction: {pred}")

    # final_prediction = gemini_prediction
    # if len(conflicting_opinions) > 1:
    #     final_prediction = verify_output(img_bytes, conflicting_opinions, mime_type)
    #     print(f"\n=== Final Prediction (after verification) ===\n{final_prediction}")
    # else:
    #     print(f"\n=== Unanimous Prediction ===\n{gemini_prediction}")
    return get_disease_analysis(gemini_prediction)