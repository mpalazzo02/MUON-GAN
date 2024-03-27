import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
_EPSILON = K.epsilon()

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import scipy

import math
import glob
import time
import shutil
import os
import random

from pickle import load
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

colours_raw_root = [[250,242,108],
					[249,225,104],
					[247,206,99],
					[239,194,94],
					[222,188,95],
					[206,183,103],
					[181,184,111],
					[157,185,120],
					[131,184,132],
					[108,181,146],
					[105,179,163],
					[97,173,176],
					[90,166,191],
					[81,158,200],
					[69,146,202],
					[56,133,207],
					[40,121,209],
					[27,110,212],
					[25,94,197],
					[34,73,162]]

colours_raw_root = np.flip(np.divide(colours_raw_root,256.),axis=0)
cmp_root = mpl.colors.ListedColormap(colours_raw_root)

print(tf.__version__)

batch_size = 50
save_interval = 250
saving_directory = 'output/'

G_architecture = [250,250,50]
D_architecture = [250,250,50]

print(' ')
print('Initializing generator network...')
print(' ')

##############################################################################################################
# Build Generative model ...
input_noise = Input(shape=(1,100))

H = Dense(int(G_architecture[0]))(input_noise)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)

H = Dense(6,activation='tanh')(H)

generator = Model(inputs=[input_noise], outputs=[H])
generator.summary()
##############################################################################################################

generator.load_weights('output/generator_weights.h5')


noise_size = 10000
									
gen_noise = np.random.normal(0, 1, (int(noise_size), 100))
images = generator.predict([np.expand_dims(gen_noise,1)])
images = np.squeeze(images)

axis_titles = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']

plt.figure(figsize=(5*4, 3*4))
subplot=0
for i in range(0, 6):
    for j in range(i+1, 6):
        subplot += 1
        plt.subplot(3,5,subplot)
        plt.hist2d(images[:noise_size,i], images[:noise_size,j], bins=50,range=[[-1,1],[-1,1]], norm=LogNorm(), cmap=cmp_root)
        plt.xlabel(axis_titles[i])
        plt.ylabel(axis_titles[j])
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('generated.png',bbox_inches='tight')
plt.close('all')