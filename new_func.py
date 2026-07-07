def irrigation_advisory(predicted_category: str) -> dict:
    """
    Maps a predicted rainfall category to an irrigation advisory.
    Handles labels like "MEDIUMRAIN", "Medium Rain", "medium rain", etc.
    """
    category = predicted_category.strip().lower().replace(" ", "")

    if category == "heavyrain":
        return {
            "level": "no_irrigation_needed",
            "message": "Heavy rainfall predicted. Irrigation not necessary this period.",
        }
    elif category == "mediumrain":
        return {
            "level": "irrigation_optional",
            "message": "Medium rainfall predicted. Irrigation may help but is not critical.",
        }
    elif category == "lightrain":
        return {
            "level": "irrigation_recommended",
            "message": "Light rainfall predicted. Irrigation recommended to supplement crop water needs.",
        }
    elif category == "norain":
        return {
            "level": "irrigation_strongly_recommended",
            "message": "No rainfall predicted. Irrigation strongly recommended.",
        }
    else:
        return {
            "level": "unknown",
            "message": f"Unrecognized rainfall category ('{predicted_category}'). Unable to provide irrigation advice.",
        }
