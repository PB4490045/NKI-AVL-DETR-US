import json
import os
from data_loading import load_fold_train_data as load_data

# Variables: change these as needed.
input_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\cocoformat\annotations2.json"       # Path to the original annotations file.
npy_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection"
output_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\cocoformat\Fold0"
fold = 0

x_train, y_train, x_val, y_val, x_test, y_test, fold_train_indices, fold_val_indices, test_indices = load_data(npy_path, fold)
                
def create_subset_annotations(set_name, indices, data):
    """
    Create a subset of the annotations data for the given indices and save it as a new JSON file.

    Args:
        set_name (str): Name of the set (e.g., "train", "val", "test").
        indices (list): List of image ids to include.
        data (dict): Original annotations data loaded from the input file.
    """
    # Copy over the keys that should remain unchanged.
    new_data = {
        "info": data.get("info"),
        "licenses": data.get("licenses"),
        "categories": data.get("categories")
    }
    
    # Filter "images" to only include those whose id is in the indices list
    # and update the "file_name" for each image.
    images_filtered = [img for img in data.get("images", []) if img.get("id") in indices]
    for img in images_filtered:
        img["file_name"] = f"{set_name}_{img.get('id')}.npy"
    new_data["images"] = images_filtered

    # Filter "annotations" to only include those whose image_id is in the indices list.
    new_data["annotations"] = [ann for ann in data.get("annotations", []) if ann.get("image_id") in indices]
    
    # Ensure the output directory exists.
    os.makedirs(output_path, exist_ok=True)
    
    # Create the output file path.
    output_file = os.path.join(output_path, f"{set_name}_annotations.json")
    
    # Save the new annotations to the file.
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Saved {set_name} annotations to {output_file}")

def main():
    # Load the original annotations file.
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Create new annotation files for each subset.
    create_subset_annotations("train", fold_train_indices, data)
    create_subset_annotations("val", fold_val_indices, data)
    create_subset_annotations("test", test_indices, data)

if __name__ == "__main__":
    main()
