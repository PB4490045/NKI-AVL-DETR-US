import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import json

"""
This will be the final script containing all pre processing functions.
"""

# def load_data(path):
# def save_data(data, path):
# def crop_images():
# def resize_images():
# def normalize_images():

def load_nparray(folder_path, index: int):
    """Load a NumPy array from a given folder and index, only considering .npy files."""
    folder_path = Path(folder_path)  # Ensure it's a Path object
    files = sorted([f for f in folder_path.iterdir() if f.suffix == '.npy'])  # Sort and filter .npy files
    
    if index >= len(files):  # Prevent index out of range errors
        raise IndexError(f"Index {index} is out of range. Folder contains {len(files)} .npy files.")
    
    np_array = np.load(files[index])
    return np_array

def prepare_dataset(x_file, y_file, test_save_prefix="test_", test_size=0.1, cv_folds=5, random_state=42):
    """
    Prepares the dataset for training an AI medical object detection model.
    
    Parameters:
        x_file (str): File path for the numpy array containing ultrasound images (e.g., "x_data.npy").
        y_file (str): File path for the numpy array containing segmentation masks (e.g., "y_data.npy").
        test_save_prefix (str): Prefix for saving the test set arrays (default "test_").
        test_size (float): Fraction of the dataset to be used as the test set (default 0.1, i.e., 10%).
        cv_folds (int): Number of folds for cross validation (default 5).
        random_state (int): Seed used to ensure reproducibility for the split.
    
    Returns:
        dict: A dictionary with keys:
            - "folds": A list of dictionaries for each CV fold with training and validation sets.
            - "x_test": The test set images.
            - "y_test": The test set segmentation masks.
    """
    # Load the arrays from the provided file paths.
    x = load_nparray(x_file, 0)
    y = load_nparray(y_file, 1)
    
    # Ensure both arrays have the same number of samples.
    assert len(x) == len(y), "The number of images and labels must match."
    
    # Generate a reproducible train-test split using indices.
    indices = np.arange(len(x))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Create the test set and save it to disk.
    x_test = x[test_idx]
    y_test = y[test_idx]
    np.save(f"{test_save_prefix}x.npy", x_test)
    np.save(f"{test_save_prefix}y.npy", y_test)
    
    # Create the training set (remaining 90% of data).
    x_train = x[train_idx]
    y_train = y[train_idx]
    
    # Prepare 5-fold cross validation splits on the training set.
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(x_train)):
        fold_data = {
            "fold": fold,
            "train_x": x_train[train_fold_idx],
            "train_y": y_train[train_fold_idx],
            "val_x": x_train[val_fold_idx],
            "val_y": y_train[val_fold_idx]
        }
        folds.append(fold_data)
    
    return {
        "folds": folds,
        "x_test": x_test,
        "y_test": y_test
    }

# Example usage:
# dataset = prepare_dataset("x_data.npy", "y_data.npy")
# Now, dataset["folds"] contains the 5-fold CV splits and the test set is saved as test_x.npy and test_y.npy.

def generate_split_indices(folder_path, output_json="split_indices.json", test_size=0.1, cv_folds=5, random_state=42):
    """
    Generates a reproducible train-test split along with cross validation folds indices
    and saves them to a JSON file.
    
    Parameters:
        x_file (str): File path for the numpy array containing ultrasound images.
        y_file (str): File path for the numpy array containing segmentation masks.
        output_json (str): File name for the JSON output containing the indices.
        test_size (float): Proportion of the data to be used as the test set (default is 0.1 for 10%).
        cv_folds (int): Number of folds for cross validation (default is 5).
        random_state (int): Seed for reproducibility.
    
    Returns:
        dict: A dictionary containing the indices for train, test, and CV folds.
    """
    # Load the numpy arrays to verify the number of samples.
    x = load_nparray(folder_path, 0)
    y = load_nparray(folder_path, 1)
    assert len(x) == len(y), "The number of images and labels must match."
    
    num_samples = len(x)
    all_indices = np.arange(num_samples)

    # Generate reproducible train-test split indices.
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Create 5-fold cross validation splits on the training indices.
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kf.split(train_indices)):
        fold_info = {
            "fold": fold_idx,
            "train_indices": train_indices[train_fold_idx].tolist(),
            "val_indices": train_indices[val_fold_idx].tolist()
        }
        folds.append(fold_info)
    
    # Create a dictionary with the indices.
    split_indices = {
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "folds": folds
    }
    
    # Save the dictionary to a JSON file in the folder_path.
    json_path = os.path.join(folder_path, output_json)
    with open(json_path, "w") as f:
        json.dump(split_indices, f, indent=4)
    
    return split_indices

    
if __name__ == "__main__":
    
    pass