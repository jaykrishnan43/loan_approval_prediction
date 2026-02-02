def apply_policy(model_proba: float, row: dict) -> float:
    r = float(row.get("emi_income_ratio", 0.0))

    # Hard rule stays
    if r > 0.7:
        return 0.0

    score = float(model_proba)

    # Soft CIBIL effect (streamlit already hard rejects < 650)
    cibil = float(row.get("cibil_score", 0))
    if cibil < 700:
        score *= 0.75
    elif cibil < 750:
        score *= 0.90

    # Stronger dependents penalty
    dep = int(row.get("no_of_dependents", 0))
    score *= max(0.40, 1.0 - 0.06 * dep)

    # Education penalty
    edu = str(row.get("education", "")).strip().lower()
    if edu == "not graduate":
        score *= 0.75

    # Extra penalty for high-but-allowed EMI ratio
    if r > 0.60:
        score *= 0.50
    elif r > 0.50:
        score *= 0.70
    elif r > 0.40:
        score *= 0.85

    # Clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score
