import numpy as np
import json
import os

# Example variables for input and output paths:
input_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection\y_collection_320_train_adjusted.npy"
output_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\cocoformat"

# Load the numpy array containing all segmentation images
segmentations = np.load(input_path)  # Expected shape: (num_images, height, width)

# Dictionary to hold annotations for each image, using image indices as keys
annotations = {}

# Define the segmentation value corresponding to tumour lesions
seg_value = 1

# Iterate through each segmentation image
for idx, seg in enumerate(segmentations):
    # Get indices where segmentation equals the target value (tumour lesion)
    indices = np.where(seg == seg_value)
    
    if indices[0].size > 0:
        # Compute bounding box: note that numpy.where returns (rows, cols)
        # so rows correspond to y-values and cols to x-values
        y_min = int(np.min(indices[0]))
        y_max = int(np.max(indices[0]))
        x_min = int(np.min(indices[1]))
        x_max = int(np.max(indices[1]))
        
        # Create the bounding box as [x_min, y_min, x_max, y_max]
        bbox = [x_min, y_min, x_max, y_max]
    else:
        # If no tumour lesion is detected, mark the bounding box as None (or you could use an empty list)
        bbox = None
    
    # Save the bounding box with the key as the image index (converted to string)
    annotations[str(idx)] = bbox

# Construct the full output path for the JSON file
json_file = os.path.join(output_path, "annotations.json")

# Save the annotations dictionary as a JSON file with pretty formatting
with open(json_file, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations saved to {json_file}")

