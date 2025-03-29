from PIL import Image
import numpy as np

# Load and preprocess the elevation map image
image_path = "N17E073.jpg"  # Ensure the correct path
img = Image.open(image_path).convert("L")  # Convert to grayscale
img = img.resize((256, 256))  # Resize to 256x256
map_array = np.array(img, dtype=np.uint8)  # Convert to numpy array
