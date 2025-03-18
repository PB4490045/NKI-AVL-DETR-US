import os
import json
import numpy as np

import preprocessing as prep

def load_fold_train_data(folder_path, fold, json_file="split_indices.json"):
    """
    Loads the training data for a specific fold using the indices stored in the JSON file.
    
    Parameters:
        folder_path (str): Folder path where the numpy arrays and JSON file are stored.
        fold (int): The fold number for which to load the training data.
        json_file (str): The name of the JSON file containing the split indices (default "split_indices.json").
    
    Returns:
        tuple: A tuple (x_train, y_train) containing the numpy arrays of the training images and labels for the specified fold.
    """
    # Build the path to the JSON file and load the split indices.
    json_path = os.path.join(folder_path, json_file)
    with open(json_path, 'r') as f:
        split_indices = json.load(f)
    
    # Find the training indices for the requested fold from the "folds" list.
    fold_train_indices = None
    fold_val_indices = None
    for fold_entry in split_indices["folds"]:
        if fold_entry["fold"] == fold:
            fold_train_indices = fold_entry["train_indices"]
            fold_val_indices = fold_entry["val_indices"]
            break
    if fold_train_indices is None:
        raise ValueError(f"Fold {fold} not found in the JSON file.")
    
    # Load the numpy arrays for images and segmentation masks.
    # Assuming load_nparray is a helper function to load the corresponding array from the folder.
    x = prep.load_nparray(folder_path, 0)
    y = prep.load_nparray(folder_path, 1)
    
    # Use the training indices for this fold to create x_train and y_train.
    x_train = x[np.array(fold_train_indices)]
    y_train = y[np.array(fold_train_indices)]
    x_val   = x[np.array(fold_val_indices)]
    y_val   = y[np.array(fold_val_indices)]
    
    return x_train, y_train, x_val, y_val

if __name__ == "__main__":
    folder_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection"
    fold = 0
    x_train, y_train = load_fold_train_data(folder_path, fold)
    print(f"Loaded training data for fold {fold} with shapes: {x_train.shape}, {y_train.shape}")

