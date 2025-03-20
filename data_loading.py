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
    test_indices = split_indices["test_indices"]
    
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
    x_test  = x[np.array(test_indices)]
    y_test  = y[np.array(test_indices)]
    
    return x_train, y_train, x_val, y_val, x_test, y_test, fold_train_indices, fold_val_indices, test_indices


def save_set(set_data, output_path, set_name, fold, indices): 
    """
    Saves individual numpy arrays with corresponding indices as separate files.
    
    Parameters:
        set_data (numpy array): Data to be saved.
        output_path (str): Path to the folder where files should be saved.
        set_name (str): Name of the dataset (e.g., "train").
        fold (int): Fold number.
        indices (list): List of indices corresponding to each saved file.
    """
    # Ensure the directory exists
    output_folder = os.path.join(output_path, f"Fold{fold}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a subfolder for the dataset
    set_folder = os.path.join(output_folder, set_name)
    os.makedirs(set_folder, exist_ok=True)
    
    # np.save(os.path.join(output_folder, f"{set_name}.npy"), set)
    # Save each numpy array individually with its corresponding index
    for i, index in enumerate(indices):
        np.save(os.path.join(set_folder, f"{set_name}_{index}.npy"), set_data[i])

if __name__ == "__main__":
    folder_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection"
    output_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\cocoformat"
    fold = 0
    x_train, y_train, x_val, y_val, x_test, y_test, fold_train_indices, fold_val_indices, test_indices = load_fold_train_data(folder_path, fold)
    print(f"Loaded training data for fold {fold} with shapes: {x_train.shape}, {y_train.shape}")
    # save_set(x_train, output_path, "train", fold, fold_train_indices)
    # save_set(x_val, output_path, "validation", fold, fold_val_indices)
    save_set(x_test, output_path, "test", fold, test_indices)

