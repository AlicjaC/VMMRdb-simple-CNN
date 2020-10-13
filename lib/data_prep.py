import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import random


def cutToLimits():

    def underLimit(limit = 100):
        cars_list = []
        for el in CLASS_NAMES_CAR:
            counter = car.count(el)
            if counter < limit:
                cars_list.append(el)
        print('\nDo usunięcia ', len(cars_list), 'samochodów.')
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

    column_names = ['path', 'make', 'model', 'year_of_production']
    raw_dataset = pd.read_csv('./cars.csv', names = column_names, sep=';',
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

    print('\nW sumie jest ', len(CLASS_NAMES_CAR), ' różnych samochodów')
    print('\nZdjęć jest ', len(train_paths))

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

    CLASS_NAMES_CAR = np.array(np.unique([name for name in car]))

    print('\nPo usunięciu < 100 zostało ', len(CLASS_NAMES_CAR), ' różnych samochodów')
    print('\nZdjęć jest ', len(train_paths))

    c = list(zip(train_paths, make, model, year))
    random.seed(4)
    random.shuffle(c)
    c = overLimit(c, 200)
    train_paths, make, model, year = zip(*c)

    train_paths = list(train_paths)
    make = list(make)
    model = list(model)
    year = list(year)

    print('\nPo usunięciu nadmiarowych zdjęć zostało ', len(train_paths))

    car = [str(make[i]) +'.'+ str(model[i]) +'.'+ str(year[i]) for i in range(len(make))]

    image_count = len(train_paths)
    
    return (train_paths, make, model, year, car)

def prepare_ds(path_list, CLASS_NAMES_MAKE, CLASS_NAMES_MODEL, CLASS_NAMES_YEAR, IMG_SIZE, one_output=False):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

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
        return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    def process_dataset(file_path):
        labels = tf.strings.split(file_path, '/')[0]
        label_make = get_label_make(labels)
        label_model = get_label_model(labels)
        label_year = get_label_year(labels)

        img = tf.io.read_file(file_path)
        img = decode_image(img)
        if one_output='make':
            return img, label_make
        elif one_output='model':
            return img, label_model
        elif one_output='year':
            return img, label_year
        else
            return img, (label_make, label_model, label_year)
    
    labeled_ds = path_list.map(process_dataset, num_parallel_calls=AUTOTUNE)
    
    return labeled_ds

#Augmentation
def augment(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tfa.image.rotate(x, tf.random.uniform(shape=[], minval=-10, maxval=10)*3.14/180)
    return x

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=512, is_training_set=False):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if is_training_set:
        ds = ds.map(lambda image, label: (augment(image), label))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
