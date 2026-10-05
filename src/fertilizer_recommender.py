"""
Fertilizer guidance for a given crop.

Design decision: there is no reliable, explainable dataset in this project
that maps soil N/P/K readings to a "correct" fertilizer type -- the ML model
that used to power this tab (models/fertilizer_model.pkl) had no training
script anywhere in the repo and its provenance couldn't be verified, so it
has been replaced with something more honest:

  1. A curated dict (agronomy reference values) for the crops most commonly
     grown in India: rice, wheat, maize, cotton, sugarcane.
  2. For any other crop in the 22-crop dataset, a DATA-DRIVEN fallback that
     reports the average N/P/K levels observed for that crop in the training
     data (models/crop_feature_stats.json) -- clearly labeled as an estimate
     from the dataset, not an agronomy recommendation.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATS_PATH = BASE_DIR / "models" / "crop_feature_stats.json"

HECTARE_TO_ACRE = 2.47105  # 1 hectare = 2.47105 acres

fertilizer_dict = {
    "rice": {
        "N": 90, "P": 40, "K": 40,
        "recommendation": "Use Urea for N, DAP for P, and MOP for K.",
        "note": "Ensure flooded conditions for rice during early growth stages.",
        "pesticide": "Carbendazim for fungal infections; insecticide for stem borer.",
        "when_to_apply": "Split N into 3 doses: 50% at transplanting, 25% at tillering, "
                          "25% at panicle initiation. Apply all P and K as basal dose at transplanting.",
        "precautions": "Wear gloves and a mask when handling Urea/DAP. Wash hands before eating. "
                        "Do not apply fertilizer right before heavy rain -- it washes away and wastes money. "
                        "Keep children and animals off the field for 24-48 hours after pesticide spray.",
    },
    "wheat": {
        "N": 100, "P": 50, "K": 50,
        "recommendation": "Apply Urea and Single Super Phosphate. MOP for K.",
        "note": "Top-dress Urea in two splits.",
        "pesticide": "Imidacloprid for aphids; foliar spray for rust control.",
        "when_to_apply": "Apply full P and K, plus half of N, as basal dose at sowing. "
                          "Top-dress the remaining N in two splits at first and second irrigation.",
        "precautions": "Avoid skin contact with concentrated Imidacloprid; wear gloves during spraying. "
                        "Don't spray on windy days. Wait at least 20 days after pesticide spray before harvest.",
    },
    "maize": {
        "N": 120, "P": 60, "K": 40,
        "recommendation": "Nitrogen from Urea; P from DAP; K from MOP.",
        "note": "Top dressing at knee-high stage.",
        "pesticide": "Chlorpyrifos for cutworms; fungicide spray for blight.",
        "when_to_apply": "Apply full P and K plus one-third N as basal dose at sowing. "
                          "Top-dress remaining N at knee-high stage and at tasseling.",
        "precautions": "Chlorpyrifos is toxic -- always wear gloves, mask, and full sleeves when spraying. "
                        "Store away from food and out of children's reach. Do not spray near water sources.",
    },
    "cotton": {
        "N": 75, "P": 40, "K": 40,
        "recommendation": "Urea for N; a complex NPK fertilizer if available.",
        "note": "Monitor for pests regularly through the season.",
        "pesticide": "Neem oil or Spinosad for bollworms and whiteflies.",
        "when_to_apply": "Apply basal dose of P and K at sowing. Split N into 2-3 doses through "
                          "the vegetative and flowering stages.",
        "precautions": "Neem oil is relatively low-risk but still wear gloves. Spray in early morning "
                        "or evening, not midday heat. Avoid spraying during flowering to protect pollinators.",
    },
    "sugarcane": {
        "N": 150, "P": 50, "K": 75,
        "recommendation": "Heavy feeder -- apply Nitrogen in split doses.",
        "note": "Add farmyard manure (FYM) during land preparation.",
        "pesticide": "Systemic pesticides for borers and white grubs.",
        "when_to_apply": "Apply full P, K, and FYM at planting. Split N into 3 doses: at planting, "
                          "at tillering (~45 days), and at grand growth stage (~90 days).",
        "precautions": "Systemic pesticides are absorbed into the plant -- always follow label dosage "
                        "exactly, do not exceed it. Wear protective gear during application and washing "
                        "of equipment afterward.",
    },
}


def to_per_acre(kg_per_ha) -> float:
    """Convert a kg/hectare figure to kg/acre -- most smallholder farmers in
    India think in acres, not hectares."""
    if not isinstance(kg_per_ha, (int, float)):
        return kg_per_ha
    return round(kg_per_ha / HECTARE_TO_ACRE, 1)


def _load_stats():
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            return json.load(f)
    return {}


_CROP_STATS = _load_stats()


def get_fertilizer_recommendation(crop_name: str) -> dict:
    crop_name = crop_name.lower().strip()

    if crop_name in fertilizer_dict:
        result = dict(fertilizer_dict[crop_name])
        result["source"] = "curated"
        return result

    stats = _CROP_STATS.get(crop_name)
    if stats:
        return {
            "N": round(stats["N"]["mean"]),
            "P": round(stats["P"]["mean"]),
            "K": round(stats["K"]["mean"]),
            "recommendation": (
                f"No curated agronomy guide exists for {crop_name} in this app yet. "
                f"Shown below are the average N/P/K levels observed for {crop_name} "
                f"in the training data -- use as a rough reference, not an exact dose."
            ),
            "note": "Consult a local agriculture extension officer for a precise dosage plan.",
            "pesticide": "Not available -- no curated pest guidance for this crop yet.",
            "when_to_apply": "Not available for this crop yet -- as a general rule, apply P and K "
                              "as a basal dose and split N across 2-3 applications through the season.",
            "precautions": "Always wear gloves and a mask when handling any fertilizer or pesticide, "
                            "wash hands before eating, and avoid application right before heavy rain.",
            "source": "data_estimate",
        }

    return {
        "N": "Unknown", "P": "Unknown", "K": "Unknown",
        "recommendation": "No recommendation available for this crop.",
        "note": "", "pesticide": "", "when_to_apply": "", "precautions": "", "source": "none",
    }
