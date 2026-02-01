def apply_policy(model_proba: float, row: dict) -> float:
    """
    model_proba: model probability of approval
    row: dict with keys: emi_income_ratio, cibil_score, no_of_dependents, education
    returns adjusted score after rules
    """

    # Hard rule
    if row["emi_income_ratio"] > 0.7:
        return 0.0

    score = float(model_proba)

    # CIBIL affects but does not dominate
    cibil = row.get("cibil_score", 0)
    if cibil < 550:
        score *= 0.35
    elif cibil < 650:
        score *= 0.70
    elif cibil < 750:
        score *= 0.92
    else:
        score *= 1.02

    # Dependents penalty
    dep = int(row.get("no_of_dependents", 0))
    score *= max(0.65, 1.0 - 0.04 * dep)

    # Education penalty
    if str(row.get("education", "")).strip().lower() == "not graduate":
        score *= 0.88

    # Clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score
