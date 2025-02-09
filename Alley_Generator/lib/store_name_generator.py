import bpy
import random
import math  # For converting degrees to radians.
from mathutils import Vector
from typing import List, Tuple, Optional, Dict

# -------------------------------------------------------------------
# Constants: Lists of descriptive words and shop types.
# -------------------------------------------------------------------
FIRST_WORDS = [
    "Flaming", "Cozy", "Golden", "Shiny", "Majestic", "Rustic", "Elegant", "Modern",
    "Happy", "Ancient", "Royal", "Bright", "Magic", "Dreamy", "Classic", "Urban",
    "Timeless", "Charming", "Bold", "Vibrant", "Lush", "Peaceful", "Witty", "Hidden",
    "Velvet", "Silent", "Playful", "Enchanted", "Serene", "Warm", "Gleaming", "Noble",
    "Festive", "Gentle", "Sunny", "Whimsical", "Lively", "Radiant", "Humble", "Pure",
    "Epic", "Grand", "Regal", "Mellow", "Sophisticated", "Trendy", "Blissful", "Glorious",
    "Red", "Blue", "Green", "Yellow", "Silver", "Copper", "Bronze", "Emerald",
    "Ruby", "Sapphire", "Crimson", "Ivory"
]

SECOND_WORDS = [
    "Grill", "Cafe", "Supplies", "Boutique", "Market", "Emporium", "Bakery", "Studio",
    "Haven", "Den", "Shop", "Corner", "Workshop", "Gallery", "Bar", "Parlor",
    "Restaurant", "Depot", "House", "Lounge", "Store", "Warehouse", "Inn", "Library",
    "Arcade", "Hall", "Retreat", "Garden", "Farm", "Oasis", "Tavern", "Pavilion",
    "Salon", "Bistro", "Deli", "Mart", "Vault", "Alcove", "Forge", "Sanctuary",
    "Grove", "Terrace", "Arena", "Dock", "Station", "Hub", "Cafe", "Barbers",
    "Bakery", "Tavern", "Boutique", "Diner", "Market", "Emporium", "Studio", "Gallery",
    "Inn", "Lounge"
]

def generate_shop_name(seed: Optional[int] = None) -> str:
    """
    Generate a random shop name using predefined word lists.
    
    Args:
        seed: An optional integer seed for reproducibility.
    
    Returns:
        A string representing the shop name.
    """
    # Use a local random generator if a seed is provided.
    rand_gen = random.Random(seed) if seed is not None else random
    first_word = rand_gen.choice(FIRST_WORDS)
    second_word = rand_gen.choice(SECOND_WORDS)
    return f"{first_word} {second_word}"

# -------------------------------------------------------------------
# Optional Testing Code
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Example usage: adjust parameters as desired.
    result = generate_shop_name(seed=12345)
    print("Shop name:", result)
