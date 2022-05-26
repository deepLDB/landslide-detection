# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:05:14 2020

@author: nagsa
"""

import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv

WORKING_DIR = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Google_deepLDB2\\IEEE_JOURNAL'
os.chdir(WORKING_DIR)

CODE_DIR = os.path.join(WORKING_DIR, 'code')
DATA_DIR = os.path.join(WORKING_DIR, 'data')
MODEL_DIR = os.path.join(WORKING_DIR, 'models')
OUTPUT_DIR = os.path.join(WORKING_DIR, 'outputs')
PRED_DIR= os.path.join(WORKING_DIR, 'predictions')

ecoregions = ['EASTERN_TEMPERATE_FORESTS', 'MARINE_WEST_COAST_FOREST',
              'NORTH_AMERICAN_DESERTS', 'NORTHWESTERN_FORESTED_MOUNTAINS']

ETF_DIR = os.path.join(DATA_DIR, ecoregions[0])
MWCF_DIR = os.path.join(DATA_DIR, ecoregions[1])
NAD_DIR = os.path.join(DATA_DIR, ecoregions[2])
NWFM_DIR = os.path.join(DATA_DIR, ecoregions[3])

folders = ['POST_IMAGES', 'POST_LABELS', 'PRE_IMAGES', 'PRE_LABELS']

file_names = {}
for i in range(len(ecoregions)):
    files = sorted(os.listdir(os.path.join(DATA_DIR, ecoregions[i], 'POST_IMAGES')))
    for f in files:
        file_names[f] = i

df = pd.DataFrame(columns=['Filenames','Class'])
df['Filenames'] = list(file_names.keys())
df['Class'] = list(file_names.values())

df.to_csv('class_names.csv', index=False, encoding='utf-8')

