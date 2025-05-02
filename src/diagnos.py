import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


def diagnose_disease(prompt_part):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    genai.configure(api_key=GEMINI_API_KEY)

    generation_config = {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 0,
        'max_output_tokens': 8192,
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config
    )
    return model.generate_content(prompt_part).text
