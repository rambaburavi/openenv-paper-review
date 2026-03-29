def grade(predicted, actual):
    
    score = 0.0

    if predicted.get("domain") == actual["domain"]:
        score += 0.4

    overlap = len(
        set(predicted.get("keywords", [])) &
        set(actual["keywords"])
    )

    score += 0.3 * (overlap / len(actual["keywords"]))

    if predicted.get("decision") == actual["decision"]:
        score += 0.3

    return round(score, 2)