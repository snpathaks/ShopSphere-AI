"""
Synthetic Data Generation Pipeline for E-Commerce Product Recommendation Engine
Generates users, products, and interaction data with realistic patterns.
"""

import numpy as np
import random
import csv
import os
from datetime import datetime, timedelta

# Reproducibility
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_USERS = 500
NUM_PRODUCTS = 200
NUM_INTERACTIONS_TARGET = 15000
START_DATE = datetime(2024, 1, 1)
DAYS_SPAN = 180

CATEGORIES = [
    "Electronics", "Clothing", "Books", "Home & Garden",
    "Sports", "Beauty", "Toys", "Food", "Automotive", "Health"
]

# Subcategories per category
SUBCATEGORIES = {
    "Electronics":   ["Headphones", "Smartphones", "Laptops", "Cameras", "Tablets", "Smart Home", "Gaming"],
    "Clothing":      ["Men's Tops", "Women's Tops", "Footwear", "Outerwear", "Activewear", "Accessories"],
    "Books":         ["Fiction", "Non-Fiction", "Science", "Programming", "History", "Self-Help", "Children's"],
    "Home & Garden": ["Furniture", "Kitchenware", "Bedding", "Garden Tools", "Lighting", "Storage"],
    "Sports":        ["Running", "Cycling", "Swimming", "Team Sports", "Fitness Equipment", "Outdoor"],
    "Beauty":        ["Skincare", "Haircare", "Makeup", "Fragrances", "Personal Care"],
    "Toys":          ["Action Figures", "Board Games", "Educational", "Dolls", "STEM", "Outdoor Toys"],
    "Food":          ["Snacks", "Beverages", "Organic", "Supplements", "Condiments", "Baking"],
    "Automotive":    ["Car Care", "Tools", "Electronics", "Accessories", "Lighting"],
    "Health":        ["Vitamins", "Medical Devices", "Fitness", "Mental Wellness", "First Aid"],
}

# Brand lists per category
BRANDS = {
    "Electronics":   ["Sony", "Samsung", "Apple", "Bose", "LG", "Anker", "JBL", "Logitech", "Razer", "Dell"],
    "Clothing":      ["Nike", "Adidas", "Levi's", "H&M", "Zara", "Under Armour", "Puma", "Gap", "Uniqlo", "Reebok"],
    "Books":         ["Penguin", "HarperCollins", "O'Reilly", "Wiley", "Scholastic", "Random House", "Packt", "Pearson"],
    "Home & Garden": ["IKEA", "KitchenAid", "Dyson", "Black+Decker", "Cuisinart", "Pottery Barn", "Home Depot", "Breville"],
    "Sports":        ["Nike", "Adidas", "Under Armour", "Callaway", "Wilson", "Speedo", "Trek", "Garmin", "New Balance"],
    "Beauty":        ["L'Oreal", "Neutrogena", "Dove", "Olay", "Estee Lauder", "CeraVe", "Maybelline", "NYX", "Clinique"],
    "Toys":          ["LEGO", "Hasbro", "Mattel", "Fisher-Price", "Melissa & Doug", "VTech", "Play-Doh", "Hot Wheels"],
    "Food":          ["Kellogg's", "Nestle", "Kind", "Clif Bar", "Larabar", "Bob's Red Mill", "Amy's", "Annie's"],
    "Automotive":    ["Armor All", "Meguiar's", "Bosch", "WeatherTech", "Chemical Guys", "Turtle Wax", "3M"],
    "Health":        ["Nature Made", "Garden of Life", "Centrum", "Omron", "Fitbit", "Thorne", "NOW Foods", "Pure Encapsulations"],
}

# Product type suffixes per subcategory
PRODUCT_TYPES = {
    "Headphones": ["Wireless Headphones", "Noise-Cancelling Headphones", "Over-Ear Headphones", "Sport Earbuds", "True Wireless Earbuds"],
    "Smartphones": ["Smartphone Pro", "Smartphone Ultra", "5G Phone", "Flagship Phone", "Budget Smartphone"],
    "Laptops": ["Laptop 15\"", "Ultrabook", "Gaming Laptop", "Business Laptop", "Chromebook"],
    "Cameras": ["DSLR Camera", "Mirrorless Camera", "Action Camera", "Webcam", "Security Camera"],
    "Tablets": ["Tablet Pro", "Drawing Tablet", "E-Reader", "Kids Tablet", "10\" Tablet"],
    "Smart Home": ["Smart Speaker", "Smart Bulb Set", "Smart Thermostat", "Smart Plug", "Security Camera"],
    "Gaming": ["Gaming Mouse", "Mechanical Keyboard", "Gaming Headset", "Controller", "Gaming Chair"],
    "Men's Tops": ["Classic T-Shirt", "Oxford Shirt", "Polo Shirt", "Flannel Shirt", "Slim-Fit Tee"],
    "Women's Tops": ["Floral Blouse", "Tank Top", "Wrap Top", "Striped Shirt", "V-Neck Sweater"],
    "Footwear": ["Running Shoes", "Casual Sneakers", "Hiking Boots", "Dress Shoes", "Sandals"],
    "Outerwear": ["Puffer Jacket", "Trench Coat", "Windbreaker", "Fleece Hoodie", "Rain Jacket"],
    "Activewear": ["Compression Leggings", "Sports Bra", "Athletic Shorts", "Training Pants", "Zip-Up Jacket"],
    "Accessories": ["Baseball Cap", "Wool Scarf", "Leather Belt", "Sunglasses", "Canvas Tote"],
    "Fiction": ["Mystery Novel", "Fantasy Novel", "Romance Novel", "Thriller", "Science Fiction Novel"],
    "Non-Fiction": ["Biography", "Memoir", "True Crime", "Essay Collection", "Journalism Book"],
    "Science": ["Physics Guide", "Chemistry Textbook", "Biology Handbook", "Astronomy Guide", "Mathematics Book"],
    "Programming": ["Python Guide", "JavaScript Handbook", "Machine Learning Book", "Data Science Guide", "Web Dev Book"],
    "History": ["World History", "Ancient Civilizations", "Modern History", "War History", "Cultural History"],
    "Self-Help": ["Productivity Guide", "Mindset Book", "Habit Building Guide", "Leadership Manual", "Finance Guide"],
    "Children's": ["Picture Book", "Chapter Book", "Activity Book", "Pop-Up Book", "Educational Book"],
    "Furniture": ["Ergonomic Chair", "Standing Desk", "Bookshelf", "Coffee Table", "Storage Ottoman"],
    "Kitchenware": ["Non-Stick Pan Set", "Knife Block Set", "Instant Pot", "Blender", "Air Fryer"],
    "Bedding": ["Microfiber Sheet Set", "Down Comforter", "Memory Foam Pillow", "Weighted Blanket", "Mattress Topper"],
    "Garden Tools": ["Garden Hose", "Pruning Shears", "Soil Mix", "Planting Kit", "Watering Can"],
    "Lighting": ["LED Desk Lamp", "Floor Lamp", "Smart Light Strip", "Pendant Light", "Table Lamp"],
    "Storage": ["Storage Bins Set", "Closet Organizer", "Under-Bed Storage", "Drawer Dividers", "Wall Shelf"],
    "Running": ["Running Shoes", "GPS Watch", "Compression Socks", "Hydration Vest", "Running Shorts"],
    "Cycling": ["Road Bike Helmet", "Cycling Gloves", "Bike Lock", "Cycling Jersey", "Bike Computer"],
    "Swimming": ["Swim Goggles", "Swim Cap", "Swim Fins", "Swimsuit", "Swim Bag"],
    "Team Sports": ["Soccer Ball", "Basketball", "Football", "Volleyball", "Baseball Glove"],
    "Fitness Equipment": ["Resistance Bands", "Dumbbell Set", "Yoga Mat", "Pull-Up Bar", "Jump Rope"],
    "Outdoor": ["Camping Tent", "Hiking Backpack", "Sleeping Bag", "Trekking Poles", "Headlamp"],
    "Skincare": ["Face Moisturizer", "Sunscreen SPF 50", "Vitamin C Serum", "Eye Cream", "Face Mask Set"],
    "Haircare": ["Shampoo", "Conditioner", "Hair Mask", "Dry Shampoo", "Hair Oil"],
    "Makeup": ["Foundation", "Mascara", "Lipstick Set", "Eyeshadow Palette", "Blush"],
    "Fragrances": ["Eau de Parfum", "Body Mist", "Cologne", "Perfume Set", "Deodorant"],
    "Personal Care": ["Electric Toothbrush", "Facial Cleanser", "Body Wash", "Razor Set", "Nail Care Kit"],
    "Action Figures": ["Superhero Figure", "Collectible Figure", "Movie Character", "Robot Toy", "Dinosaur Set"],
    "Board Games": ["Strategy Game", "Family Card Game", "Trivia Game", "Cooperative Game", "Party Game"],
    "Educational": ["Math Flashcards", "Alphabet Puzzle", "Science Kit", "Globe", "Memory Game"],
    "Dolls": ["Fashion Doll", "Baby Doll", "Plush Toy", "Paper Doll Set", "Doll Accessories"],
    "STEM": ["Coding Kit", "Electronics Kit", "Robot Building Set", "Chemistry Set", "Engineering Blocks"],
    "Outdoor Toys": ["Water Blaster", "Jump Rope Set", "Frisbee", "Kite", "Bubbles Kit"],
    "Snacks": ["Mixed Nuts", "Protein Bar", "Granola", "Popcorn", "Trail Mix"],
    "Beverages": ["Green Tea Pack", "Coffee Blend", "Protein Shake Mix", "Electrolyte Drink", "Herbal Tea"],
    "Organic": ["Organic Honey", "Organic Oats", "Organic Pasta", "Organic Olive Oil", "Organic Quinoa"],
    "Supplements": ["Whey Protein", "Creatine Powder", "BCAA Supplement", "Pre-Workout", "Collagen Peptides"],
    "Condiments": ["Hot Sauce", "BBQ Sauce", "Sriracha", "Salsa", "Pesto Sauce"],
    "Baking": ["All-Purpose Flour", "Baking Powder", "Vanilla Extract", "Chocolate Chips", "Almond Flour"],
    "Car Care": ["Car Wash Kit", "Microfiber Towels", "Tire Inflator", "Dashboard Cleaner", "Wax Polish"],
    "Tools": ["Socket Set", "Torque Wrench", "Jump Starter", "OBD Scanner", "Floor Jack"],
    "Automotive Electronics": ["Dash Cam", "Bluetooth Car Kit", "GPS Navigator", "Car Phone Mount", "Backup Camera"],
    "Automotive Accessories": ["Seat Covers", "Floor Mats", "Steering Wheel Cover", "Car Organizer", "Sunshade"],
    "Automotive Lighting": ["LED Headlights", "Interior Lights", "LED Strip", "Fog Lights", "Tail Lights"],
    "Vitamins": ["Vitamin D3", "Vitamin C", "B-Complex", "Zinc Supplement", "Magnesium"],
    "Medical Devices": ["Blood Pressure Monitor", "Pulse Oximeter", "Thermometer", "Glucose Monitor", "TENS Unit"],
    "Fitness": ["Foam Roller", "Massage Gun", "Resistance Bands", "Stability Ball", "Acupressure Mat"],
    "Mental Wellness": ["Meditation App Kit", "Stress Relief Journal", "Essential Oil Diffuser", "Sleep Mask", "CBD Gummies"],
    "First Aid": ["First Aid Kit", "Bandage Assortment", "Antiseptic Spray", "Cold Pack", "Medical Tape"],
}

# Description templates
DESCRIPTION_TEMPLATES = [
    "High-quality {product} with premium materials and excellent performance for everyday use by all ages.",
    "Experience the best in class {product} featuring advanced technology and sleek modern design.",
    "Durable and reliable {product} built to last with top-notch craftsmanship and superior comfort.",
    "Innovative {product} designed for maximum efficiency and ease of use in any situation.",
    "Premium {product} offering exceptional value with cutting-edge features and long-lasting durability.",
    "Ergonomic {product} engineered for comfort and performance, perfect for daily activities and adventures.",
    "Award-winning {product} trusted by thousands of customers for its outstanding quality and reliability.",
    "Versatile {product} suitable for beginners and experts alike, backed by stellar customer reviews.",
    "Compact and lightweight {product} delivering powerful performance in a convenient portable package.",
    "Feature-rich {product} packed with everything you need to get the most out of your experience.",
    "Eco-friendly {product} made with sustainable materials while maintaining exceptional quality standards.",
    "Best-selling {product} loved by millions worldwide for its unmatched quality and affordable price.",
    "Professional-grade {product} used by experts and enthusiasts for superior results every time.",
    "Stylish and functional {product} that combines modern aesthetics with practical everyday usability.",
    "Top-rated {product} with rave reviews from satisfied customers who love its performance and design.",
]

USER_SEGMENTS = ["budget", "mid_range", "premium", "luxury"]
GENDERS = ["M", "F", "Non-binary"]
US_LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
    "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
    "Fort Worth, TX", "Columbus, OH", "Charlotte, NC", "Indianapolis, IN",
    "San Francisco, CA", "Seattle, WA", "Denver, CO", "Nashville, TN",
    "Oklahoma City, OK", "El Paso, TX", "Boston, MA", "Portland, OR",
    "Las Vegas, NV", "Memphis, TN", "Louisville, KY", "Baltimore, MD",
    "Milwaukee, WI", "Albuquerque, NM",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fmt_user_id(n):
    return f"U{n:04d}"

def fmt_item_id(n):
    return f"I{n:04d}"

def fmt_session_id(n):
    return f"S{n:07d}"

def fmt_interaction_id(n):
    return f"INT{n:08d}"


# ─────────────────────────────────────────────
# Generate Users
# ─────────────────────────────────────────────

def generate_users(n=NUM_USERS):
    users = []
    signup_start = datetime(2020, 1, 1)
    signup_end = datetime(2023, 12, 31)
    signup_range = (signup_end - signup_start).days

    for i in range(1, n + 1):
        age = int(np.random.randint(18, 70))
        gender = random.choices(GENDERS, weights=[0.48, 0.48, 0.04])[0]
        location = random.choice(US_LOCATIONS)
        days_offset = int(np.random.randint(0, signup_range))
        signup_date = (signup_start + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        segment = random.choices(USER_SEGMENTS, weights=[0.35, 0.40, 0.18, 0.07])[0]
        users.append({
            "user_id": fmt_user_id(i),
            "age": age,
            "gender": gender,
            "location": location,
            "signup_date": signup_date,
            "user_segment": segment,
        })
    return users


# ─────────────────────────────────────────────
# Generate Products
# ─────────────────────────────────────────────

def generate_products(n=NUM_PRODUCTS):
    products = []
    used_names = set()
    # Distribute products across categories
    per_cat = n // len(CATEGORIES)
    remainder = n % len(CATEGORIES)

    item_num = 1
    for cat_idx, category in enumerate(CATEGORIES):
        count = per_cat + (1 if cat_idx < remainder else 0)
        subcats = SUBCATEGORIES[category]
        brands = BRANDS[category]

        for _ in range(count):
            subcat = random.choice(subcats)
            brand = random.choice(brands)

            # Build product name
            type_key = subcat
            # Try to get product type from PRODUCT_TYPES dict
            type_list = PRODUCT_TYPES.get(subcat, [f"{subcat} Product"])
            suffix = random.choice(type_list)

            # Add a model number/year variant to ensure uniqueness
            model_variants = [
                "", " Pro", " Plus", " Max", " Lite", " Ultra", " SE",
                " 2024", " Gen 2", " v2", " Elite", " Premium", " Classic",
            ]
            attempt = 0
            while True:
                mv = random.choice(model_variants) if attempt > 0 else ""
                name = f"{brand} {suffix}{mv}".strip()
                if name not in used_names:
                    used_names.add(name)
                    break
                attempt += 1
                if attempt > 50:
                    name = f"{brand} {suffix} #{item_num}"
                    used_names.add(name)
                    break

            # Price ranges per category
            price_ranges = {
                "Electronics": (29.99, 1499.99),
                "Clothing": (9.99, 299.99),
                "Books": (5.99, 59.99),
                "Home & Garden": (12.99, 599.99),
                "Sports": (9.99, 499.99),
                "Beauty": (4.99, 149.99),
                "Toys": (7.99, 199.99),
                "Food": (3.99, 59.99),
                "Automotive": (8.99, 299.99),
                "Health": (6.99, 199.99),
            }
            lo, hi = price_ranges.get(category, (9.99, 199.99))
            price = round(random.uniform(lo, hi), 2)

            avg_rating = round(random.uniform(3.0, 5.0), 1)
            num_reviews = int(np.random.randint(5, 5000))

            # Generate description (15-30 words)
            tmpl = random.choice(DESCRIPTION_TEMPLATES)
            description = tmpl.format(product=name)

            products.append({
                "item_id": fmt_item_id(item_num),
                "name": name,
                "category": category,
                "subcategory": subcat,
                "price": price,
                "description": description,
                "brand": brand,
                "avg_rating": avg_rating,
                "num_reviews": num_reviews,
            })
            item_num += 1

    return products


# ─────────────────────────────────────────────
# Latent vectors
# ─────────────────────────────────────────────

def build_latent_vectors():
    """Return user (500x10) and item (200x10) latent matrices."""
    user_latent = np.random.randn(NUM_USERS, 10)
    item_latent = np.random.randn(NUM_PRODUCTS, 10)
    return user_latent, item_latent


# ─────────────────────────────────────────────
# Popularity weights (Zipf / power-law)
# ─────────────────────────────────────────────

def build_popularity_weights():
    """Power-law popularity: top 20% items get ~80% traffic."""
    ranks = np.arange(1, NUM_PRODUCTS + 1, dtype=float)
    weights = 1.0 / (ranks ** 1.2)   # Zipf exponent
    weights /= weights.sum()
    return weights


# ─────────────────────────────────────────────
# Realistic timestamp generation
# ─────────────────────────────────────────────

def random_timestamp():
    """
    Business-hours weighted timestamp within a 180-day window.
    Weekdays weighted 3x over weekends.
    Hours 9-21 weighted 4x over off-hours.
    """
    day_offset = int(np.random.choice(
        DAYS_SPAN,
        p=_day_weights()
    ))
    ts = START_DATE + timedelta(days=day_offset)

    # Weighted hour
    hour = int(np.random.choice(24, p=_hour_weights()))
    minute = int(np.random.randint(0, 60))
    second = int(np.random.randint(0, 60))
    ts = ts.replace(hour=hour, minute=minute, second=second)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _day_weights():
    """Weekday vs weekend weights over 180 days starting 2024-01-01."""
    weights = []
    for d in range(DAYS_SPAN):
        day = (START_DATE + timedelta(days=d)).weekday()
        weights.append(3.0 if day < 5 else 1.0)
    arr = np.array(weights)
    return arr / arr.sum()


def _hour_weights():
    weights = np.ones(24)
    weights[9:22] = 4.0   # business / active hours
    return weights / weights.sum()


# Pre-compute once
_DAY_WEIGHTS = _day_weights()
_HOUR_WEIGHTS = _hour_weights()


def fast_timestamp():
    day_offset = int(np.random.choice(DAYS_SPAN, p=_DAY_WEIGHTS))
    ts = START_DATE + timedelta(days=day_offset)
    hour = int(np.random.choice(24, p=_HOUR_WEIGHTS))
    minute = int(np.random.randint(0, 60))
    second = int(np.random.randint(0, 60))
    ts = ts.replace(hour=hour, minute=minute, second=second)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────
# Generate Interactions
# ─────────────────────────────────────────────

INTERACTION_TYPES = ["view", "cart", "purchase"]
INTERACTION_WEIGHTS = [1, 3, 5]   # relative probability ratios used below
# In practice: view >> cart >> purchase realistically
INTERACTION_PROBS = [0.65, 0.22, 0.13]

def generate_interactions(users, products, user_latent, item_latent, pop_weights,
                           target=NUM_INTERACTIONS_TARGET):
    """
    Generate interaction records.
    - Latent dot-product boosts probability of a user picking certain items.
    - Popularity (Zipf) is the primary item-selection distribution.
    - For each sampled (user, item) pair we assign an interaction type.
    - Ratings only for 'purchase' interactions.
    """
    interactions = []
    session_counter = 1
    int_counter = 1

    user_ids = [u["user_id"] for u in users]
    item_ids = [p["item_id"] for p in products]

    # Normalised dot product scores per user (500 x 200)
    scores = user_latent @ item_latent.T          # (500, 200)
    # Softmax per user to get probability adjustment
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    user_item_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (500, 200)

    # Blend latent probs with popularity weights: 40% latent, 60% popularity
    blended = 0.4 * user_item_probs + 0.6 * pop_weights[np.newaxis, :]
    blended /= blended.sum(axis=1, keepdims=True)

    # Group interactions into sessions (2-8 interactions per session)
    while len(interactions) < target:
        # Pick a random user
        u_idx = int(np.random.randint(0, NUM_USERS))
        user_id = user_ids[u_idx]
        session_id = fmt_session_id(session_counter)
        session_counter += 1

        session_len = int(np.random.randint(2, 9))
        # Shared session timestamp base (all events in a session within 1 hour)
        base_ts = fast_timestamp()
        base_dt = datetime.strptime(base_ts, "%Y-%m-%d %H:%M:%S")

        for _ in range(session_len):
            if len(interactions) >= target + 1000:
                break
            # Sample item using blended probabilities
            item_idx = int(np.random.choice(NUM_PRODUCTS, p=blended[u_idx]))
            item_id = item_ids[item_idx]

            # Interaction type
            itype = random.choices(INTERACTION_TYPES, weights=INTERACTION_PROBS)[0]

            # Offset within session
            offset_secs = int(np.random.randint(0, 3600))
            evt_dt = base_dt + timedelta(seconds=offset_secs)
            # Keep within span
            if evt_dt > START_DATE + timedelta(days=DAYS_SPAN):
                evt_dt = base_dt
            timestamp = evt_dt.strftime("%Y-%m-%d %H:%M:%S")

            # Rating only for purchases
            rating = None
            if itype == "purchase":
                rating = round(random.choices(
                    [1, 2, 3, 4, 5],
                    weights=[0.03, 0.07, 0.15, 0.35, 0.40]
                )[0] * 1.0, 1)

            interactions.append({
                "interaction_id": fmt_interaction_id(int_counter),
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": itype,
                "timestamp": timestamp,
                "rating": rating if rating is not None else "",
                "session_id": session_id,
            })
            int_counter += 1

        if len(interactions) >= target:
            break

    return interactions


# ─────────────────────────────────────────────
# Write CSVs
# ─────────────────────────────────────────────

def write_csv(filepath, fieldnames, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {filepath}")


# ─────────────────────────────────────────────
# Data Summary
# ─────────────────────────────────────────────

def write_summary(users, products, interactions):
    path = os.path.join(OUTPUT_DIR, "data_summary.txt")

    # Basic stats
    ages = [u["age"] for u in users]
    prices = [p["price"] for p in products]
    ratings_num = [float(i["rating"]) for i in interactions if i["rating"] != ""]

    itype_counts = {}
    for i in interactions:
        itype_counts[i["interaction_type"]] = itype_counts.get(i["interaction_type"], 0) + 1

    cat_counts = {}
    for p in products:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("E-COMMERCE RECOMMENDATION ENGINE — DATA SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"USERS ({len(users):,} total)\n")
        f.write(f"  Age range  : {min(ages)} – {max(ages)}\n")
        f.write(f"  Avg age    : {sum(ages)/len(ages):.1f}\n")
        segments = {}
        for u in users:
            segments[u["user_segment"]] = segments.get(u["user_segment"], 0) + 1
        for seg, cnt in sorted(segments.items()):
            f.write(f"  Segment '{seg}': {cnt}\n")

        f.write(f"\nPRODUCTS ({len(products):,} total)\n")
        f.write(f"  Price range: ${min(prices):.2f} – ${max(prices):.2f}\n")
        f.write(f"  Avg price  : ${sum(prices)/len(prices):.2f}\n")
        f.write("  Category distribution:\n")
        for cat, cnt in sorted(cat_counts.items()):
            f.write(f"    {cat:<20}: {cnt}\n")

        f.write(f"\nINTERACTIONS ({len(interactions):,} total)\n")
        f.write(f"  Date span  : {START_DATE.date()} → {(START_DATE + timedelta(days=DAYS_SPAN)).date()}\n")
        for itype, cnt in sorted(itype_counts.items()):
            pct = cnt / len(interactions) * 100
            f.write(f"  {itype:<10}: {cnt:,} ({pct:.1f}%)\n")

        if ratings_num:
            f.write(f"\n  Ratings (purchase only):\n")
            f.write(f"    Count    : {len(ratings_num):,}\n")
            f.write(f"    Avg      : {sum(ratings_num)/len(ratings_num):.2f}\n")
            f.write(f"    Min/Max  : {min(ratings_num):.1f} / {max(ratings_num):.1f}\n")

    print(f"  Wrote summary → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E-Commerce Data Generation Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating users ...")
    users = generate_users()
    write_csv(
        os.path.join(OUTPUT_DIR, "users.csv"),
        ["user_id", "age", "gender", "location", "signup_date", "user_segment"],
        users,
    )

    print("\n[2/5] Generating products ...")
    products = generate_products()
    write_csv(
        os.path.join(OUTPUT_DIR, "products.csv"),
        ["item_id", "name", "category", "subcategory", "price", "description", "brand", "avg_rating", "num_reviews"],
        products,
    )

    print("\n[3/5] Building latent vectors & popularity weights ...")
    user_latent, item_latent = build_latent_vectors()
    pop_weights = build_popularity_weights()
    print(f"  User latent: {user_latent.shape}, Item latent: {item_latent.shape}")
    print(f"  Pop weights sum: {pop_weights.sum():.4f}")

    print("\n[4/5] Generating interactions ...")
    interactions = generate_interactions(users, products, user_latent, item_latent, pop_weights)
    write_csv(
        os.path.join(OUTPUT_DIR, "interactions.csv"),
        ["interaction_id", "user_id", "item_id", "interaction_type", "timestamp", "rating", "session_id"],
        interactions,
    )

    print("\n[5/5] Writing data summary ...")
    write_summary(users, products, interactions)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  users.csv       : {len(users):,} rows")
    print(f"  products.csv    : {len(products):,} rows")
    print(f"  interactions.csv: {len(interactions):,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
