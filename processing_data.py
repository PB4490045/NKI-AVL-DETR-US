from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_nparray(folder_path: Path, index: int):
    """Load a NumPy array from a given folder and index."""
    folder_path = Path(folder_path)  # Ensure it's a Path object
    files = sorted(folder_path.iterdir())  # Sort to ensure consistent ordering
    
    if index >= len(files):  # Prevent index out of range errors
        raise IndexError(f"Index {index} is out of range. Folder contains {len(files)} files.")
    
    np_array = np.load(files[index])
    return np_array

def visualize_data_index(np_array, index):
    """Visualize a specific index of a NumPy array."""
    plt.imshow(np_array[index])
    plt.show()

if __name__ == "__main__":
    # Define needed paths
    data_path = Path(r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection")
    
    # Create numpy arrays of data
    us_frames = load_nparray(data_path, 0)
    segmentations = load_nparray(data_path, 1)
