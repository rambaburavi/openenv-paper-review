def grade(predicted, actual):
    
    score = 0.0

    # Domain match
    if predicted.get("domain") == actual["domain"]:
        score += 0.4

    elif predicted.get("domain", "").lower() in actual["domain"].lower():
        score += 0.2


    # Keyword overlap
    overlap = len(
        set(predicted.get("keywords", [])) &
        set(actual["keywords"])
    )

    keyword_ratio = overlap / max(len(actual["keywords"]), 1)

    score += 0.3 * keyword_ratio


    # Decision match
    if predicted.get("decision") == actual["decision"]:
        score += 0.3


    # Confidence calibration bonus
    confidence = predicted.get("confidence")

    if confidence is not None:
        if 0.6 <= confidence <= 0.9:
            score += 0.02


    # Difficulty scaling
    difficulty_weights = {
        "easy": 0.95,
        "medium": 1.0,
        "hard": 1.05
    }

    score *= difficulty_weights.get(
        actual.get("difficulty", "medium"),
        1.0
    )


    # Clamp validator-safe
    if score <= 0:
        score = 0.01

    elif score >= 1:
        score = 0.99


    return round(score, 4)