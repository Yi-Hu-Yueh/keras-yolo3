#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)
    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, model_to_save):
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 5, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5', 
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )    
    return [early_stop, checkpoint, reduce_on_plateau]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh,  
    saved_weights_name, 
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale  
):
    train_model, infer_model = create_yolov3_model(
        nb_class            = nb_class, 
        anchors             = anchors, 
        max_box_per_image   = max_box_per_image, 
        max_grid            = max_grid, 
        batch_size          = batch_size, 
        warmup_batches      = warmup_batches,
        ignore_thresh       = ignore_thresh,
        grid_scales         = grid_scales,
        obj_scale           = obj_scale,
        noobj_scale         = noobj_scale,
        xywh_scale          = xywh_scale,
        class_scale         = class_scale
    )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        train_model.load_weights(saved_weights_name)
    else:
        train_model.load_weights("backend.h5", by_name=True)          

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

# Parse the annotations 
train_annot_folder = './kangaroo/annots/'
train_image_folder = './kangaroo/images/'
cache_name = 'kangaroo_train.pkl'
valid_annot_folder = ''
valid_image_folder = ''
valid_cache_name = ''
labels = ["kangaroo"]

train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
    train_annot_folder,
    train_image_folder,
    cache_name,
    valid_annot_folder,
    valid_image_folder,
    valid_cache_name,
    labels
)
print('\nTraining on: \t' + str(labels) + '\n')

# Create the generators 
anchors = [55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]
batch_size = 4
max_input_size = 448
min_input_size = 288

train_generator = BatchGenerator(
    instances           = train_ints, 
    anchors             = anchors,   
    labels              = labels,        
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = batch_size,
    min_net_size        = min_input_size,
    max_net_size        = max_input_size,   
    shuffle             = True, 
    jitter              = 0.3, 
    norm                = normalize
)

valid_generator = BatchGenerator(
    instances           = valid_ints, 
    anchors             = anchors,   
    labels              = labels,        
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = batch_size,
    min_net_size        = min_input_size,
    max_net_size        = max_input_size,   
    shuffle             = True, 
    jitter              = 0.0, 
    norm                = normalize
)

# Create the model 
saved_weights_name = 'kangaroo.h5'
warmup_epochs = 3
train_times = 8
ignore_thresh = 0.5
learning_rate = 1e-4
grid_scales = [1,1,1]
obj_scale = 5
if os.path.exists(saved_weights_name): 
    warmup_epochs = 0
warmup_batches = warmup_epochs * (train_times*len(train_generator))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_model, infer_model = create_model(
    nb_class            = len(labels), 
    anchors             = anchors, 
    max_box_per_image   = max_box_per_image, 
    max_grid            = [max_input_size, max_input_size], 
    batch_size          = batch_size, 
    warmup_batches      = warmup_batches,
    ignore_thresh       = 0.5,
    saved_weights_name  = saved_weights_name,
    lr                  = 1e-4,
    grid_scales         = [1,1,1],
    obj_scale           = 5,
    noobj_scale         = 1,
    xywh_scale          = 1,
    class_scale         = 1,
)

# Start training
callbacks = create_callbacks(saved_weights_name, infer_model)
nb_epochs = 100
train_model.fit_generator(
    generator        = train_generator, 
    steps_per_epoch  = len(train_generator) * train_times, 
    epochs           = nb_epochs + warmup_epochs, 
    verbose          = 2,
    callbacks        = callbacks, 
    workers          = 4,
    max_queue_size   = 8
)

# Run the evaluation
# compute mAP for all the classes
average_precisions = evaluate(infer_model, valid_generator)

# print the score
for label, average_precision in average_precisions.items():
    print(labels[label] + ': {:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           
