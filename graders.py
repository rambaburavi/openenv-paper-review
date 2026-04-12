import random


def grade(predicted, actual):

    score = 0.0

    # Domain match (core relevance)
    if predicted.get("domain") == actual["domain"]:
        score += 0.4
    elif predicted.get("domain", "").lower() in actual["domain"].lower():
        score += 0.2  # partial semantic tolerance


    # Keyword overlap (partial progress reward)
    overlap = len(
        set(predicted.get("keywords", [])) &
        set(actual["keywords"])
    )

    keyword_ratio = overlap / max(len(actual["keywords"]), 1)
    score += 0.3 * keyword_ratio


    # Decision match (accept/reject recommendation)
    if predicted.get("decision") == actual["decision"]:
        score += 0.3


    # Difficulty-aware scaling (curriculum learning signal)
    difficulty_weights = {
        "easy": 0.9,
        "medium": 1.0,
        "hard": 1.1
    }

    score *= difficulty_weights.get(actual["difficulty"], 1.0)


    # Add small stochastic noise (simulates reviewer variance)
    score += random.uniform(-0.02, 0.02)


    # Clamp strictly inside (0, 1) — required by validator
    score = min(max(score, 0.01), 0.99)


    return score