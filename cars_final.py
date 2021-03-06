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
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorboard.plugins.hparams import api as hp

import optuna
from optuna.samplers import CmaEsSampler

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

labeled_ds = data_prep.prepare_ds(path_list, CLASS_NAMES_MAKE, CLASS_NAMES_MODEL, CLASS_NAMES_YEAR, IMG_WIDTH)

train_size = math.ceil(0.7 * len(train_paths))
train_ds = labeled_ds.take(train_size)
val_ds = labeled_ds.skip(train_size)

train_set = data_prep.prepare_for_training(train_ds, batch_size = BATCH_SIZE, is_training_set=True)
val_set = data_prep.prepare_for_training(val_ds, batch_size = BATCH_SIZE)

mirrored_strategy = tf.distribute.MirroredStrategy()

def params_search():
    with mirrored_strategy.scope():
        
        neuron_1 = 156
        kernel_1 = 5
        neuron_2 = 312
        kernel_2 = 3
        neuron_3 = 624
        kernel_3 = 3        
        neuron_4 = 1248
        kernel_4 = 3
        
        image_input = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='input_image')
        x = layers.Conv2D(neuron_1, (kernel_1, kernel_1), use_bias=False)(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_2, (kernel_2, kernel_2), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_3, (kernel_3, kernel_3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_4, (kernel_4, kernel_4), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        num_dense_make = 5
        make_neuron = 612
        num_dense_model = 1
        model_neuron = 1429
        num_dense_year = 3
        year_neuron = 433
        
        l2_parametr_make = 7e-6
        l2_make = tf.keras.regularizers.l2(l2_parametr_make)
        make_dropout = 0.2       
        
        l2_parametr_model = 4.4e-6
        l2_model = tf.keras.regularizers.l2(l2_parametr_model)
        model_dropout = 0.35
        
        l2_parametr_year = 9e-6
        l2_year = tf.keras.regularizers.l2(l2_parametr_year)
        year_dropout = 0.25

        l_make = layers.Dense(make_neuron, activation='relu', kernel_regularizer = l2_make)(x)
        l_model = layers.Dense(model_neuron, activation='relu', kernel_regularizer = l2_model)(x)
        l_year = layers.Dense(year_neuron, activation='relu', kernel_regularizer = l2_year)(x)
   
        for i in range(int(num_dense_make-1)):
            l_make = layers.Dense(make_neuron, activation='relu', kernel_regularizer = l2_make)(l_make)
        l_make = layers.Dropout(make_dropout)(l_make)

        for i in range(int(num_dense_model-1)):
            l_model = layers.Dense(model_neuron, activation='relu', kernel_regularizer = l2_model)(l_model)
        l_model = layers.Dropout(model_dropout)(l_model)
        
        for i in range(int(num_dense_year-1)):
            l_year = layers.Dense(year_neuron, activation='relu', kernel_regularizer = l2_year)(l_year)
        l_year = layers.Dropout(year_dropout)(l_year)

        output_make = layers.Dense(len(CLASS_NAMES_MAKE), activation='softmax', name='output_make')(l_make)
        output_model = layers.Dense(len(CLASS_NAMES_MODEL), activation='softmax', name='output_model')(l_model)
        output_year = layers.Dense(len(CLASS_NAMES_YEAR), activation='softmax', name='output_year')(l_year)

        cnn = keras.Model(inputs=image_input, outputs=[output_make, output_model, output_year], name='cars_model')

        #or if you want to use pretrained model:
#         cnn = tf.keras.models.load_model(path_to_model)
        cnn.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    loss_weights=[1., 1., 1.],
                    metrics=['acc'])

    
    #CALLBACKS
    logdir = logs_dir + datetime.datetime.now().strftime("%b-%d-%H%M")
    tbcall = tf.keras.callbacks.TensorBoard(logdir, profile_batch=0)
    
    checkpoint_filepath = models_dir + 'model_checkpoint.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_make_acc',
        mode='max',
        save_best_only=True)
    
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
            epochs = 1000,
            callbacks=[tbcall, model_checkpoint_callback],
            validation_data = val_set,
            validation_steps = STEPS_PER_EPOCH_VAL
            )
    return cnn

model = params_search()

model.save(models_dir + datetime.datetime.now().strftime("%b-%d-%H%M") + '.h5')
