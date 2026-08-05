import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Provider registry: everything that differs between backends lives here ---
PROVIDERS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key":  "MISTRAL_API_TOKEN",
        "headers":  {},
    },
    "apertus": {
        "base_url": "https://api.publicai.co/v1",
        "env_key":  "PUBLICAI_API_TOKEN",
        "headers":  {"User-Agent": "ttcb-classifier/1.0"},  # required by Public AI
    },
}

_clients = {}  # cache one client per provider

def get_client(provider):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDERS)}.")
    if provider not in _clients:
        cfg = PROVIDERS[provider]
        token = os.getenv(cfg["env_key"])
        if token is None:
            raise ValueError(f"{cfg['env_key']} not found in .env for provider '{provider}'.")
        _clients[provider] = OpenAI(
            base_url=cfg["base_url"],
            api_key=token,
            default_headers=cfg["headers"],
        )
    return _clients[provider]


def classify_project(title, description, context, role, tool="TT",
                     provider="mistral", model="mistral-small-latest",
                     delay=1.5, max_retries=5):
    """
    Classify one project via the chosen provider. Returns "1", "0", or None.
    None = genuine API/parse failure (never coerced to 0).
    """
    client = get_client(provider)
    prompt = context + f"Project Title: {title}\nProject Description: {description}"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0,
                stream=False,
            )
            output = response.choices[0].message.content.strip()
            time.sleep(delay)
            if output.startswith("1"):
                return "1"
            if output.startswith("0"):
                return "0"
            print(f"⚠️ Unparseable {tool} output for '{title}': '{output}'")
            return None
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = 2 ** attempt
                print(f"  ⏳ rate limited, waiting {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"Error classifying {tool} for '{title}': {e}")
            return None
    return None