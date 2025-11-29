from __future__ import annotations

from typing import Optional, Tuple

Canonical = Tuple[str | None, str]

# Canonical vocabularies
SCENES = {
    "indoor",
    "outdoor",
    "street",
    "park",
    "yard",
    "garden",
    "beach",
    "stadium",
    "museum",
    "restaurant",
    "cafe",
    "store",
    "kitchen",
    "living room",
    "dining room",
    "bedroom",
    "office",
    "school",
    "classroom",
    "gym",
    "church",
    "concert",
    "parking lot",
}

OBJECTS = {
    "person",
    "child",
    "adult",
    "baby",
    "pet",
    "dog",
    "cat",
    "bird",
    "car",
    "bicycle",
    "stroller",
    "tree",
    "plant",
    "flower",
    "building",
    "table",
    "counter",
    "chair",
    "couch",
    "sofa",
    "bed",
    "pillow",
    "blanket",
    "bench",
    "window",
    "door",
    "stairs",
    "wall",
    "floor",
    "fridge",
    "refrigerator",
    "oven",
    "stove",
    "microwave",
    "sink",
    "toilet",
    "bathtub",
    "phone",
    "laptop",
    "computer",
    "monitor",
    "tv",
    "book",
    "notebook",
    "tablet",
    "camera",
    "gift",
    "cake",
    "cupcake",
    "balloon",
    "food",
    "drink",
    "pizza",
    "ice cream",
    "cookie",
    "drink",
    "bottle",
    "cup",
    "glass",
    "plate",
    "backpack",
    "purse",
    "bag",
    "hat",
    "coat",
    "jacket",
    "shirt",
    "dress",
    "sweater",
    "pants",
    "jeans",
    "shoes",
    "boots",
    "scarf",
    "gloves",
    "mask",
    "christmas tree",
    "wreath",
    "ornament",
    "lights",
    "candle",
    "sign",
    "banner",
    "toy",
    "doll",
    "stuffed animal",
    "ball",
}

ACTIVITIES = {
    "posing",
    "smiling",
    "hugging",
    "holding hands",
    "eating",
    "cooking",
    "drinking",
    "playing",
    "watching tv",
    "reading",
    "writing",
    "studying",
    "drawing",
    "painting",
    "working",
    "using computer",
    "using phone",
    "working",
    "driving",
    "walking",
    "running",
    "riding bicycle",
    "shopping",
    "celebrating",
    "dancing",
    "sitting",
    "standing",
}

EVENTS = {
    "birthday",
    "wedding",
    "holiday",
    "christmas",
    "halloween",
    "easter",
    "graduation",
    "party",
    "concert",
    "sports event",
    "family gathering",
    "vacation",
    "picnic",
}

COLORS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
}

BRANDS = {"nike", "adidas", "apple", "samsung", "google", "sony"}
TIME_OF_DAY = {"morning", "afternoon", "evening", "night"}
WEATHER = {"sunny", "cloudy", "rainy", "snowy", "foggy"}
AGE_BANDS = {"child", "teen", "adult", "senior"}
GENDERS = {"male", "female", "unknown"}
PEOPLE_ATTR = {"glasses", "hat", "mask", "smiling", "beard"}
QUALITY = {"blur-high", "blur-medium", "blur-low"}
BUCKETS = {
    "people",
    "groups_events",
    "selfie",
    "pets_animals",
    "food_recipe",
    "documents",
    "notes_handwriting",
    "receipts_bills",
    "screenshots",
    "diagrams_charts",
    "maps_navigation",
    "memes_comics",
    "art_illustration",
    "objects_items",
    "landscapes_outdoors",
    "vehicles_transport",
    "home_interiors",
    "screens_displays",
    "shopping_products",
    "misc_other",
}

# Cross-category synonyms -> (category, canonical value)
SYNONYMS: dict[str, Canonical] = {
    # scenes
    "indoors": ("scene", "indoor"),
    "inside": ("scene", "indoor"),
    "outside": ("scene", "outdoor"),
    "livingroom": ("scene", "living room"),
    "class room": ("scene", "classroom"),
    "parking": ("scene", "parking lot"),
    "backyard": ("scene", "yard"),
    "garden area": ("scene", "garden"),
    # objects
    "people": ("object", "person"),
    "woman": ("object", "adult"),
    "man": ("object", "adult"),
    "boy": ("object", "child"),
    "girl": ("object", "child"),
    "kid": ("object", "child"),
    "kids": ("object", "child"),
    "tv screen": ("object", "tv"),
    "television": ("object", "tv"),
    "pc": ("object", "computer"),
    "desktop": ("object", "computer"),
    "cell phone": ("object", "phone"),
    "mobile phone": ("object", "phone"),
    "smartphone": ("object", "phone"),
    "luggage": ("object", "bag"),
    "suitcase": ("object", "bag"),
    "handbag": ("object", "purse"),
    "snow": ("scene", "outdoor"),
    "snowy": ("scene", "outdoor"),
    "refrigerator": ("object", "fridge"),
    "freezer": ("object", "fridge"),
    "microwave oven": ("object", "microwave"),
    "stovetop": ("object", "stove"),
    # activities
    "taking photo": ("activity", "posing"),
    "taking picture": ("activity", "posing"),
    "photo shoot": ("activity", "posing"),
    "watching television": ("activity", "watching tv"),
    "watching tv": ("activity", "watching tv"),
    "use phone": ("activity", "using phone"),
    "using phone": ("activity", "using phone"),
    "working on laptop": ("activity", "working"),
    # events
    "birthday party": ("event", "birthday"),
    "christmas party": ("event", "christmas"),
    "halloween party": ("event", "halloween"),
    "sports game": ("event", "sports event"),
    # time/weather
    "sunset": ("time", "evening"),
    "sunrise": ("time", "morning"),
}


def map_label(prefix: Optional[str], value: str) -> Optional[Canonical]:
    """Map (prefix, value) into a canonical (category, label) in the taxonomy."""
    v = value.strip().lower()
    if not v:
        return None

    # Try synonyms first
    if v in SYNONYMS:
        cat, val = SYNONYMS[v]
        if prefix and prefix.replace("_", "-") != cat:
            return None
        return cat, val

    cat = prefix.replace("_", "-") if prefix else None

    if cat == "scene":
        if v in SCENES:
            return "scene", v
    elif cat == "object":
        if v in OBJECTS:
            return "object", v
    elif cat == "activity":
        if v in ACTIVITIES:
            return "activity", v
    elif cat == "event":
        if v in EVENTS:
            return "event", v
    elif cat == "color":
        if v in COLORS:
            return "color", v
    elif cat == "brand":
        if v in BRANDS:
            return "brand", v
    elif cat == "time":
        if v in TIME_OF_DAY:
            return "time", v
    elif cat == "weather":
        if v in WEATHER:
            return "weather", v
    elif cat == "age-band":
        if v in AGE_BANDS:
            return "age-band", v
    elif cat == "gender":
        if v in GENDERS:
            return "gender", v
    elif cat == "people-attr":
        if v in PEOPLE_ATTR:
            return "people-attr", v
    elif cat == "pets-count":
        # numeric allowed
        try:
            int(v)
            return "pets-count", v
        except ValueError:
            return None
    elif cat == "people-count":
        try:
            int(v)
            return "people-count", v
        except ValueError:
            return None
    elif cat == "quality":
        if v in QUALITY:
            return "quality", v
    elif cat == "bucket":
        candidate = v.replace(" ", "_").replace("-", "_")
        if candidate in BUCKETS:
            return "bucket", candidate

    # If no prefix, allow limited unprefixed tags for compatibility.
    if cat is None:
        # Try to infer category by membership in vocab sets (priority order).
        if v in SCENES:
            return "scene", v
        if v in OBJECTS:
            return "object", v
        if v in ACTIVITIES:
            return "activity", v
        if v in EVENTS:
            return "event", v
        if v in COLORS:
            return "color", v
        if v in BRANDS:
            return "brand", v
        if v in TIME_OF_DAY:
            return "time", v
        if v in WEATHER:
            return "weather", v
        if v in {"portrait", "selfie", "group", "family", "pet", "child", "adult"}:
            return None, v
    return None


def taxonomy_labels(include_people_and_pets: bool = True) -> list[str]:
    """
    Return canonical label strings with prefixes for embedding/scoring.

    Example: ["scene:indoor", "object:tree", "activity:cooking", "bucket:people"]
    """
    labels: set[str] = set()

    labels.update(f"scene:{s}" for s in SCENES)
    labels.update(f"activity:{a}" for a in ACTIVITIES)
    labels.update(f"event:{e}" for e in EVENTS)
    labels.update(f"color:{c}" for c in COLORS)
    labels.update(f"brand:{b}" for b in BRANDS)
    labels.update(f"time:{t}" for t in TIME_OF_DAY)
    labels.update(f"weather:{w}" for w in WEATHER)
    labels.update(f"quality:{q}" for q in QUALITY)
    labels.update(f"bucket:{b}" for b in BUCKETS)

    for obj in OBJECTS:
        if not include_people_and_pets and obj in {
            "person",
            "adult",
            "child",
            "baby",
            "pet",
            "dog",
            "cat",
            "bird",
        }:
            continue
        labels.add(f"object:{obj}")

    return sorted(labels)
