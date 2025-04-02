import numpy as np
import json
import os
from datetime import datetime

# Define your input and output paths (update these paths accordingly)
# input_path: path to your .npy file containing segmentation arrays.
# output_path: directory where "annotations.json" will be saved.
# Example:
input_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection\y_collection_320_train_adjusted.npy"
output_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\cocoformat"
 
# Load the NumPy array of segmentation images.
segmentations = np.load(input_path)  # Expected shape: (num_images, height, width)

# Create the base COCO-style dictionary.
coco_output = {
    "info": {
        "year": "2025",  # or use str(datetime.now().year) if you prefer dynamic year
        "version": "1",
        "description": "annotations of US images",
        "contributors": "Bart van den Berg",
        "url": "",
        "date_created": "19-03-2025"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "healthy",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "tumour lesion",
            "supercategory": "tumour"
        }
    ],
    "images": [],
    "annotations": []
}

annotation_id = 0

# Process each segmentation
for idx, seg in enumerate(segmentations):
    # Get image dimensions from the segmentation array.
    height, width = seg.shape
    
    # Create an image entry (file_name and date_captured are not needed here).
    image_entry = {
        "id": idx,
        "license": 1,
        "file_name": idx,  
        "height": height,
        "width": width,
        "date_captured": ""
    }
    coco_output["images"].append(image_entry)
    
    # Identify tumour lesion pixels (assumed to be labelled with value 1)
    indices = np.where(seg == 1)
    if indices[0].size > 0:
        # Compute bounding box coordinates
        y_min = int(np.min(indices[0]))
        y_max = int(np.max(indices[0]))
        x_min = int(np.min(indices[1]))
        x_max = int(np.max(indices[1]))
        
        # Compute width and height of the bounding box
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Create annotation entry in COCO format.
        annotation_entry = {
            "id": annotation_id,
            "image_id": idx,
            "category_id": 1,  # All bounding boxes for tumour lesions.
            "bbox": [x_min, y_min, x_max, y_max], 
            "area": bbox_width * bbox_height,
            "segmentation": [],  # Empty list as in the attached file.
            "iscrowd": 0
        }
        coco_output["annotations"].append(annotation_entry)
        annotation_id += 1

# Save the output JSON file in the output path.
output_file = os.path.join(output_path, "annotations2.json")
with open(output_file, "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"Annotations saved to {output_file}")
