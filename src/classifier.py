import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API token securely from environment variable
load_dotenv()
api_token = os.getenv("HF_API_TOKEN")

if api_token is None:
    raise ValueError("Hugging Face API token not found. Make sure it's set in your .env file.")

# Set up OpenAI-compatible Hugging Face endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/novita/v3/openai",
    api_key=api_token,
)

def classify_project(title, description, context, role, tool="TT", delay=0.5):
    """
    Calls the Hugging Face API to classify a single project based on context and role.

    Args:
        title (str): Project title.
        description (str): Project description.
        context (str): User prompt context with definition + examples.
        role (str): System role prompt.
        tool (str): "TT" or "CB" (used for debugging/logging).
        delay (float): Optional delay to avoid rate limits.

    Returns:
        str: Model response ("1" or "0")
    """
    prompt = (
        context +
        f"Project Title: {title}\n"
        f"Project Description: {description}"
    )
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.2,
            stream=False,
        )
        output = response.choices[0].message.content.strip()

        # Optional: enforce numeric output
        # if output not in {"0", "1"}:
        #    print(f"⚠️ Unexpected output for {tool}: '{output}' — defaulting to '0'")
        #    return "0"

        return output

    except Exception as e:
        print(f"Error while classifying {tool} for project '{title}': {e}")
        return "0"
    finally:
        time.sleep(delay)
