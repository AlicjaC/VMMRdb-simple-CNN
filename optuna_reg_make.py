#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import math
import random
try:
    import cPickle as pickle
except:
    import pickle
import joblib

import sys
sys.path.insert(1, './lib')
import data_prep
import lr_warmup

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorboard.plugins.hparams import api as hp

import optuna
from optuna.samplers import TPESampler

from functools import partial

dataset_dir = './Car_Dataset'
logs_dir = './logs/'
models_dir = './models/'

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

tf.config.optimizer.set_jit(True)

train_paths, make, model, year, car = data_prep.cutToLimits()
multiplier = 8
BATCH_SIZE = 32 * multiplier
IMG_WIDTH = 96
IMG_HEIGHT = 96
STEPS_PER_EPOCH = np.ceil(len(train_paths)/BATCH_SIZE)
STEPS_PER_EPOCH_TRAIN = np.ceil(len(train_paths)/BATCH_SIZE*0.7)
STEPS_PER_EPOCH_VAL = np.ceil(len(train_paths)/BATCH_SIZE*0.3)

CLASS_NAMES_MAKE = np.array(np.unique([name for name in make]))
CLASS_NAMES_MODEL = np.array(np.unique([name for name in model]))
CLASS_NAMES_YEAR = np.array(np.unique([name for name in year]))
CLASS_NAMES_CAR = np.array(np.unique([name for name in car]))

os.chdir(dataset_dir)

AUTOTUNE = tf.data.experimental.AUTOTUNE

path_list = tf.data.Dataset.from_tensor_slices(train_paths)

labeled_ds = data_prep.prepare_ds(path_list, CLASS_NAMES_MAKE, CLASS_NAMES_MODEL, CLASS_NAMES_YEAR, IMG_WIDTH, one_output='make')

train_size = math.ceil(0.7 * len(train_paths))
train_ds = labeled_ds.take(train_size)
val_ds = labeled_ds.skip(train_size)

train_set = data_prep.prepare_for_training(train_ds, batch_size = BATCH_SIZE)
val_set = data_prep.prepare_for_training(val_ds, batch_size = BATCH_SIZE)

mirrored_strategy = tf.distribute.MirroredStrategy()

def params_search(trial):
    with mirrored_strategy.scope():
        neuron_1 = 457
        kernel_1 = 7 
        neuron_2 = 934
        kernel_2 = 7 
        neuron_3 = 471 
        kernel_3 = 5 
        neuron_4 = 1343 
        kernel_4 = 5 

        conv_dropout = trial.suggest_discrete_uniform('conv_dropout', 0.0, 0.5, 0.05)
        
        image_input = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='input_image')
        x = layers.Conv2D(neuron_1, (kernel_1, kernel_1), use_bias=False, name='conv_1')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_2, (kernel_2, kernel_2), activation='relu', name='conv_2')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_3, (kernel_3, kernel_3), activation='relu', name='conv_3')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_4, (kernel_4, kernel_4), activation='relu', name='conv_4')(x)
        x = layers.GlobalAveragePooling2D()(x)

        l2_make = tf.keras.regularizers.l2(trial.suggest_loguniform('l2_make', 1e-8, 1e-4))

        num_dense_make = 9
        make_neuron = 1563
        make_dropout = trial.suggest_discrete_uniform('make_dropout', 0.0, 0.5, 0.05)

        l_make = layers.Dense(make_neuron, activation='relu', kernel_regularizer = l2_make)(x)

        for i in range(int(num_dense_make-1)):
            l_make = layers.Dense(make_neuron, activation='relu', kernel_regularizer = l2_make)(l_make)
        l_make = layers.Dropout(make_dropout)(l_make)

        output_make = layers.Dense(len(CLASS_NAMES_MAKE), activation='softmax', name='output_make')(l_make)

        cnn = keras.Model(inputs=image_input, outputs=output_make)

        cnn.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])
        
    cnn.load_weights('weights.h5', by_name=True)
        
    #CALLBACKS
    sample_count = np.ceil(len(train_paths)*0.7)
    epochs = 1000
    warmup_epoch = 10
    learning_rate_base = 4e-4

    total_steps = int(epochs * sample_count / BATCH_SIZE)
    warmup_steps = int(warmup_epoch * sample_count / BATCH_SIZE)
    warmup_batches = warmup_epoch * sample_count / BATCH_SIZE

    warm_up_lr = lr_warmup.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)

    cnn.fit(train_set,
            steps_per_epoch = STEPS_PER_EPOCH_TRAIN,
            epochs = 40,
            callbacks=[warm_up_lr],
            validation_data = val_set,
            validation_steps = STEPS_PER_EPOCH_VAL
            )
    
    loss, make_acc = cnn.evaluate(val_set, steps=STEPS_PER_EPOCH_VAL)
    
    return loss


study = optuna.create_study(sampler = TPESampler(n_startup_trials=100))

study.optimize(params_search, n_trials=120)

joblib.dump(study, logs_dir + 'make_reg120.pkl')
