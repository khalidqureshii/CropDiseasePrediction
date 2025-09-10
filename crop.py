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
Common Root Rot, Fusarium Head Blight, Mildew, Mite, Septoria, Smut, Stem Fly, Tan Spot, Yellow Rust"""



def analyze_disease(img_bytes, mime_type):
    prompt = f"""
        You are an agricultural expert. Identify the crop and disease in this image. Ignore watermarks or any text at corners.

        Crops: {crop_options}
        Diseases: {disease_options}

        Return ONLY valid JSON in this exact format:
        {{
            "crop": "<name>",
            "disease": "<name>",
            "causes": ["cause1", "cause2", "cause3"],
            "recommendations": ["rec1", "rec2", "rec3"]
        }}
        Do not include any text outside the JSON.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]
        }]
    )
    return response.text.strip()


