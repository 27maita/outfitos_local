import os
import cv2
import json
import random
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
from PIL import Image

project_folder = os.path.dirname(os.path.abspath(__file__))  
local_storage_path = os.path.join(project_folder, "closet")  
closet_file_path = os.path.join(project_folder, "closet_data.json")  
detected_items_file_path = os.path.join(project_folder, "detected_items.json")  

os.makedirs(local_storage_path, exist_ok=True)

model = YOLO('deepfashion2_yolov8s-seg.pt')

def extract_dominant_colors(image_path, k=3):
    img = Image.open(image_path).convert('RGB') 
    img_np = np.array(img)
    img_reshaped = img_np.reshape((-1, 3))  # Reshape to 2D array
    kmeans = KMeans(n_clusters=k).fit(img_reshaped)  # Perform KMeans clustering
    dominant_colors = np.round(kmeans.cluster_centers_).astype(int)      return dominant_colors

def rgb_to_name(rgb, threshold=100):
    min_distance = float('inf')
    closest_color = 'unknown'
    for color_name, color_rgb in mcolors.CSS4_COLORS.items():
        hex_rgb = tuple(int(c * 255) for c in mcolors.hex2color(color_rgb))  
        distance = np.sqrt(np.sum((np.array(rgb) - np.array(hex_rgb)) ** 2))
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_color = color_name
    return closest_color

def classify_item(item_name):
    tops_keywords = ["shirt", "blouse", "jacket", "sweater"]
    bottoms_keywords = ["pants", "shorts", "skirt", "jeans"]
    if any(keyword in item_name.lower() for keyword in tops_keywords):
        return "top"
    elif any(keyword in item_name.lower() for keyword in bottoms_keywords):
        return "bottom"
    else:
        return "unknown"

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)

    detected_items = []

    if results[0].masks.data.shape[0] > 0:
        for i, mask in enumerate(results[0].masks.data):
            item_name = results[0].names[results[0].boxes.cls[i].cpu().item()]
            confidence = results[0].boxes.conf[i].cpu().item()

            mask = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != image_rgb.shape[:2]:
                mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

            clothing_item = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = min(x_indices), max(x_indices)
            y_min, y_max = min(y_indices), max(y_indices)
            cropped_clothing = clothing_item[y_min:y_max, x_min:x_max]

            base_filename = f"{item_name}.png"
            output_path = os.path.join(local_storage_path, base_filename)
            counter = 1
            while os.path.exists(output_path):
                base_filename = f"{item_name}_{counter}.png"
                output_path = os.path.join(local_storage_path, base_filename)
                counter += 1

            cv2.imwrite(output_path, cv2.cvtColor(cropped_clothing, cv2.COLOR_RGB2BGR))
            detected_items.append(base_filename)
            print(f"Saved {item_name} at {output_path} with confidence {confidence:.2%}")

    with open(detected_items_file_path, 'w') as f:
        json.dump(detected_items, f)

    return detected_items

def update_closet(detected_items):
    if os.path.exists(closet_file_path):
        with open(closet_file_path, 'r') as f:
            closet = json.load(f)
    else:
        closet = {}

    for item in detected_items:
        item_path = os.path.join(local_storage_path, item)
        if os.path.exists(item_path):
            category = classify_item(item)
            dominant_colors = extract_dominant_colors(item_path)
            color_names = [rgb_to_name(tuple(color)) for color in dominant_colors]

            closet[item] = {
                "filename": item,
                "colors": color_names,
                "style": [],  # Empty style list for now
                "category": category
            }
            print(f"Added {item} to closet dictionary with category '{category}' and colors {color_names}.")

    with open(closet_file_path, 'w') as f:
        json.dump(closet, f, indent=4)

    print("Closet updated.")

def load_closet():
    if os.path.exists(closet_file_path):
        with open(closet_file_path, 'r') as f:
            return json.load(f)
    return {}

COLOR_MATCHES = {
    "black": ["white", "gray", "red", "blue", "green", "pink"],
    "white": ["black", "blue", "red", "pink", "green"],
    "gray": ["black", "white", "blue", "red"],
    "red": ["black", "white", "gray", "blue"],
    "blue": ["black", "white", "gray", "red", "yellow"],
    "green": ["black", "white", "pink", "yellow"],
    "yellow": ["blue", "green", "black"],
    "pink": ["black", "white", "green", "gray"]
}

def recommend_outfit():
    closet = load_closet()
    tops = [item for item in closet.values() if item["category"] == "top"]
    bottoms = [item for item in closet.values() if item["category"] == "bottom"]

    if not tops or not bottoms:
        print("Not enough items in the closet to make an outfit.")
        return None

    outfits = []
    for top in tops:
        for bottom in bottoms:
            for top_color in top["colors"]:
                for bottom_color in bottom["colors"]:
                    if top_color in COLOR_MATCHES and bottom_color in COLOR_MATCHES[top_color]:
                        outfits.append((top["filename"], bottom["filename"]))
                        break  # Stop checking once a match is found

    if outfits:
        recommended_outfit = random.choice(outfits)
    else:
        print("No matching outfits found based on color combinations. Choosing a random outfit.")
        recommended_outfit = (random.choice(tops)["filename"], random.choice(bottoms)["filename"])

    print(f"Recommended Outfit: {recommended_outfit}")
    return recommended_outfit

def main():
    image_path = 'clothes/0a1955a3-c4c5-4141-8a72-afbcf92d0dfa.jpg'  
    detected_items = process_image(image_path)
    update_closet(detected_items)
    recommend_outfit()

if __name__ == "__main__":
    main()