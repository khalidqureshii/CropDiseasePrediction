from IPython.display import Markdown, display # type: ignore
from google import genai
from openai import OpenAI # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import re
import json

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# --- Options ---
crop_options = "Wheat, Mustard, Potato"

disease_options = """Healthy, Aphid, Black Rust, Brown Rust, Blast Test, Leaf Blight,
Common Root Rot, Fusarium Head Blight, Mildew, Mite, Septoria, Smut, Stem Fly, Tan Spot, Yellow Rust"""



def analyze_disease(img_bytes, mime_type):
    prompt = f"""
        You are an agricultural expert. Identify the crop and disease in this image. Ignore watermarks or any text at corners.

        Crops: {crop_options}
        Diseases: {disease_options}

        Return ONLY valid JSON in this exact format, in ONE SINGLE LINE (no line breaks, no markdown, no extra text):
        {{"crop": "<name>", "disease": "<name>", "causes": ["cause1", "cause2"], "recommendations": ["rec1", "rec2"]}}
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]
        }]
    )
    raw_text = response.text.strip()

    # --- Step 1: Strip markdown fences if present ---
    cleaned_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()

    # --- Step 2: Remove line breaks ---
    cleaned_text = cleaned_text.replace("\n", "").replace("\r", "").strip()

    # --- Step 3: Extract JSON object if extra text sneaks in ---
    match = re.search(r"\{.*\}", cleaned_text)
    if not match:
        raise ValueError(f"Invalid response format: {raw_text}")
    
    json_str = match.group(0)

    # --- Step 4: Parse JSON safely ---
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {json_str}") from e

    return result


