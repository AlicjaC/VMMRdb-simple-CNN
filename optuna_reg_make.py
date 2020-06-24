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

tf.config.optimizer.set_jit(True)

def underLimit(limit = 100):
    cars_list = []
    for el in CLASS_NAMES_CAR:
        counter = car.count(el)
        if counter < limit:
            cars_list.append(el)
    return cars_list

def overLimit(lista, limit = 200):
    for el in CLASS_NAMES_CAR:
        if car.count(el) > 200:
            counter = 0
            deleted = 0
            for i in range(len(lista)):
                if ((str(lista[i-deleted][1]) +'.'+ str(lista[i-deleted][2]) +'.'+ str(lista[i-deleted][3])) == el):
                    counter += 1
                    if counter > 200:
                        del lista[i-deleted]
                        deleted += 1
    return lista

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
            
column_names = ['path', 'make', 'model', 'year_of_production']
raw_dataset = pd.read_csv('/home/ubuntu/Zadanie_3/cars.csv', names = column_names, sep=';',
                      dtype={"path": str, "make": str, 'model': str, 'year_of_production': str})

raw_dataset = raw_dataset[raw_dataset.path != 'path']
raw_dataset = raw_dataset[raw_dataset.year_of_production != 'ints']

train_paths = raw_dataset['path'].values.tolist()
make = raw_dataset['make'].values.tolist()
model = raw_dataset['model'].values.tolist()
year = raw_dataset['year_of_production'].values.tolist()

car = [str(make[i]) +'.'+ str(model[i]) +'.'+ str(year[i]) for i in range(len(make))]

CLASS_NAMES_MAKE = np.array(np.unique([name for name in make]))
CLASS_NAMES_MODEL = np.array(np.unique([name for name in model]))
CLASS_NAMES_YEAR = np.array(np.unique([name for name in year]))
CLASS_NAMES_CAR = np.array(np.unique([name for name in car]))

cars_list = underLimit(100)

deleted_counter = 0
for i in range(len(train_paths)):
    if car[i-deleted_counter] in cars_list:
        del train_paths[i-deleted_counter]
        del make[i-deleted_counter]
        del model[i-deleted_counter]
        del year[i-deleted_counter]
        del car[i-deleted_counter]
        deleted_counter += 1
    
def fConstant ():
    return 0.1
c = list(zip(train_paths, make, model, year))
random.shuffle(c, fConstant)
c = overLimit(c, 200)
train_paths, make, model, year = zip(*c)

train_paths = list(train_paths)
make = list(make)
model = list(model)
year = list(year)

car = [str(make[i]) +'.'+ str(model[i]) +'.'+ str(year[i]) for i in range(len(make))]

image_count = len(train_paths)

multiplier = 16 
BATCH_SIZE = 32 * multiplier
IMG_WIDTH = 96 
IMG_HEIGHT = 96 
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
STEPS_PER_EPOCH_TRAIN = np.ceil(image_count/BATCH_SIZE*0.7)
STEPS_PER_EPOCH_VAL = np.ceil(image_count/BATCH_SIZE*0.3)

CLASS_NAMES_MAKE = np.array(np.unique([name for name in make]))
CLASS_NAMES_MODEL = np.array(np.unique([name for name in model]))
CLASS_NAMES_YEAR = np.array(np.unique([name for name in year]))
CLASS_NAMES_CAR = np.array(np.unique([name for name in car]))

os.chdir('Car_Dataset')

AUTOTUNE = tf.data.experimental.AUTOTUNE

path_list = tf.data.Dataset.from_tensor_slices(train_paths)

def get_label_make(labels):
    make = tf.strings.split(labels, '_')[0]
    return make == CLASS_NAMES_MAKE
def get_label_model(labels):
    pos = tf.strings.length(tf.strings.split(labels, '_')[0])+1
    length = tf.strings.length(labels)-pos-5
    model = tf.strings.substr(labels, pos, length)
    return model == CLASS_NAMES_MODEL
def get_label_year(labels):
    year = tf.strings.split(labels, '_')[-1]
    return year == CLASS_NAMES_YEAR

def decode_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img*2-1
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_dataset(file_path):
    labels = tf.strings.split(file_path, '/')[0]
    label_make = get_label_make(labels)
    label_model = get_label_model(labels)
    label_year = get_label_year(labels)
     
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, label_make

labeled_ds = path_list.map(process_dataset, num_parallel_calls=AUTOTUNE)

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_size = math.ceil(0.7 * image_count)
train_ds = labeled_ds.take(train_size)
test_ds = labeled_ds.skip(train_size)

train_set = prepare_for_training(train_ds)
test_set = prepare_for_training(test_ds)

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
        x = layers.Conv2D(neuron_1, (kernel_1, kernel_1), use_bias=False, name='conv_1', trainable=False)(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_2, (kernel_2, kernel_2), activation='relu', name='conv_2', trainable=False)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_3, (kernel_3, kernel_3), activation='relu', name='conv_3', trainable=False)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(neuron_4, (kernel_4, kernel_4), activation='relu', name='conv_4', trainable=False)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(conv_dropout)(x)
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
    sample_count = np.ceil(image_count*0.7)
    epochs = 1000
    warmup_epoch = 10
    learning_rate_base = 4e-4

    total_steps = int(epochs * sample_count / BATCH_SIZE)
    warmup_steps = int(warmup_epoch * sample_count / BATCH_SIZE)
    warmup_batches = warmup_epoch * sample_count / BATCH_SIZE

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)

    cnn.fit(train_set,
            steps_per_epoch = STEPS_PER_EPOCH_TRAIN,
            epochs = 40,
            callbacks=[warm_up_lr],
            validation_data = test_set,
            validation_steps = STEPS_PER_EPOCH_VAL
            )
    
    loss, make_acc = cnn.evaluate(test_set, steps=STEPS_PER_EPOCH_VAL)
    
    return loss


study = optuna.create_study(sampler = TPESampler(n_startup_trials=100))

study.optimize(params_search, n_trials=120)

joblib.dump(study, 'optuna_trials/make_reg120.pkl')
