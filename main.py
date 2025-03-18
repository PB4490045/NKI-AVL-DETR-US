import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

import preprocessing as prep
import postprocessing as pop



if __name__ == "__main__":
    
    #Define needed paths
    folder_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection"
    
    prep.generate_split_indices(folder_path)
   
   
   
   
    #Create numpy arrays of data

    # unique, counts = np.unique(segmentations, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(us_frames[0])
    # #Visualize data
    # plt.figure(figsize=(20, 10))
    # plt.subplot(2, 4, 1)
    # plt.imshow(us_frames[12], cmap='gray')
    # plt.subplot(2, 4, 2)
    # plt.imshow(us_frames[33], cmap='gray')
    # plt.subplot(2, 4, 3)
    # plt.imshow(us_frames[500], cmap='gray')
    # plt.subplot(2, 4, 4)
    # plt.imshow(us_frames[1000], cmap='gray')
    # plt.subplot(2, 4, 5)
    # plt.imshow(segmentations[12], cmap='gray')
    # plt.subplot(2, 4, 6)
    # plt.imshow(segmentations[33], cmap='gray')
    # plt.subplot(2, 4, 7)
    # plt.imshow(segmentations[500], cmap='gray')
    # plt.subplot(2, 4, 8)
    # plt.imshow(segmentations[1000], cmap='gray')
    # plt.show()
    
    
    
  