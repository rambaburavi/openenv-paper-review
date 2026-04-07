def grade(predicted, actual):
    
    score = 0.0

    # Domain match
    if predicted.get("domain") == actual["domain"]:
        score += 0.4

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

    # Ensure score is strictly between (0, 1)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return round(score, 2)