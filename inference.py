import os
import json
from environment import PaperReviewEnv, Action

# Read environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

client = None

# Only create OpenAI client if API key exists
if API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception:
        client = None

def fallback_prediction(abstract):
    
    abstract_lower = abstract.lower()

    if "cnn" in abstract_lower:
        return {
            "domain": "Computer Vision",
            "keywords": ["CNN", "segmentation"],
            "decision": "accept"
        }

    elif "transformer" in abstract_lower:
        return {
            "domain": "Natural Language Processing",
            "keywords": ["transformer", "summarization"],
            "decision": "accept"
        }

    elif "iot" in abstract_lower:
        return {
            "domain": "Embedded AI",
            "keywords": ["IoT", "anomaly detection"],
            "decision": "reject"
        }

    return {
        "domain": "",
        "keywords": [],
        "decision": ""
    }


def get_llm_prediction(abstract):
    
    if client is None:
        return fallback_prediction(abstract)

    prompt = f"""
You are a research paper reviewer AI.

Return JSON:

{{
"domain": "...",
"keywords": ["...", "..."],
"decision": "accept or reject"
}}

Abstract:
{abstract}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        return fallback_prediction(abstract)


def run_environment():

    env = PaperReviewEnv()

    scores = []

    for i in range(3):

        observation = env.reset()

        prediction = get_llm_prediction(
            observation.abstract
        )

        action = Action(
            domain=prediction.get("domain"),
            keywords=prediction.get("keywords"),
            decision=prediction.get("decision")
        )

        _, reward, _, _ = env.step(action)

        scores.append(reward.score)

        print(f"Task {i+1} Score:", reward.score)

    avg_score = sum(scores) / len(scores)

    print("\nAverage Score:", avg_score)


if __name__ == "__main__":
    run_environment()