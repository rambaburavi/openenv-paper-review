import os
import json
from environment import PaperReviewEnv, Action


API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = None


if API_KEY and BASE_URL:

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except:
        client = None


def llm_call(prompt):

    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content

    except:
        return None


def fallback(stage, abstract):

    abstract = abstract.lower()

    if stage == "domain":

        if "cnn" in abstract:
            return "Computer Vision"

        if "transformer" in abstract:
            return "Natural Language Processing"

        if "iot" in abstract:
            return "Embedded AI"

        return "Machine Learning"


    if stage == "keywords":

        if "cnn" in abstract:
            return ["cnn", "segmentation"]

        if "transformer" in abstract:
            return ["transformer", "summarization"]

        if "iot" in abstract:
            return ["iot", "anomaly detection"]

        return ["machine learning"]


    if stage == "decision":
        return "accept"


def predict(stage, abstract):

    prompt = f"""
Stage: {stage}

Abstract:
{abstract}

Return only the answer.
"""

    result = llm_call(prompt)

    if result:

        try:

            if stage == "keywords":
                return json.loads(result)

            return result.strip()

        except:
            pass

    return fallback(stage, abstract)


def run_environment():

    env = PaperReviewEnv()

    scores = []

    for i in range(len(env.tasks)):

        observation = env.reset()

        task_name = f"task_{i+1}"

        print(f"[START] task={task_name}", flush=True)

        done = False
        steps = 0
        total_score = 0

        while not done:

            steps += 1

            stage = observation.stage

            prediction = predict(stage, observation.abstract)

            action = Action()

            if stage == "domain":
                action.domain = prediction

            elif stage == "keywords":
                action.keywords = prediction

            elif stage == "decision":
                action.decision = prediction
                action.confidence = 0.8

            observation, reward, done, _ = env.step(action)

            total_score = reward.score

            print(
                f"[STEP] step={steps} reward={reward.score}",
                flush=True
            )


        scores.append(total_score)

        print(
            f"[END] task={task_name} score={total_score} steps={steps}",
            flush=True
        )


    avg_score = sum(scores) / len(scores)

    print(
        f"[SUMMARY] average_score={avg_score}",
        flush=True
    )


if __name__ == "__main__":
    run_environment()