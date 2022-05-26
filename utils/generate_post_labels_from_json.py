# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:45:51 2020

@author: nagsa
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
import subprocess
import shutil

path_to_wd = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Google_deepLDB2\\deepLDB\\IEEE_JOURNAL'
WORKING_DIR = os.path.join(path_to_wd)

DATA_DIR = os.path.join(WORKING_DIR, 'data')
CODE_DIR = os.path.join(WORKING_DIR, 'code')

os.chdir(CODE_DIR)

ecoregions = ['EASTERN_TEMPERATE_FORESTS', 
              'MARINE_WEST_COAST_FOREST',
              'NORTH_AMERICAN_DESERTS',
              'NORTHWESTERN_FORESTED_MOUNTAINS']

folder_names = ['POST_IMAGES',
                'POST_JSON_LABELS',
                'POST_LABELS']


path_to_labelme_file = 'C:\\Users\\nagsa\\Anaconda3\\Scripts\\labelme_json_to_dataset.exe'

# use it once and comment it back
'''
for i in range(len(ecoregions)):
    path_to_labels = os.path.join(DATA_DIR, ecoregions[i], folder_names[1])
    files = os.listdir(path_to_labels)
    
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
        
    for j in range(len(files)):    
        subprocess.run([path_to_labelme_file, os.path.join(path_to_labels, files[j]), '-o', os.path.join(DATA_DIR, ecoregions[i], folder_names[2], files[j][:-5])])
        print(j+1)
'''
for i in range(len(ecoregions)):
    path_to_labels = os.path.join(DATA_DIR, ecoregions[i], folder_names[1])
    files = os.listdir(path_to_labels)
    img_files = os.listdir(os.path.join(DATA_DIR, ecoregions[i], folder_names[0]))

    if 'desktop.ini' in files:
        files.remove('desktop.ini')
        
    path = os.path.join(DATA_DIR, ecoregions[i], folder_names[2])
    folders = os.listdir(path)
    for j in range(len(folders)):
        p = os.path.join(path, folders[j], 'label.png')
        shutil.copy(p, path)
        os.rename(os.path.join(path, 'label.png'), os.path.join(path, img_files[j]))
        
    
    