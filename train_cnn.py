import cv2
import numpy as np
import argparse

from sklearn.utils import class_weight

from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, Conv2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
#############################################    

fixed_width_position = 112
fixed_height_position = 224

def load_data(image_folder, filepath):
    gt_file = open(filepath, 'r')
    gt_list = gt_file.read().splitlines()
    gt_file.close()

    X_shape = []
    Y_shape = []
    X_position = []
    Y_position = []

    # Open images with dynamic programming
    images = {}

    for line in gt_list:
        if line[0] == '#':
            continue
        else:
            # 00518.JPG ; 1 ; ; 1 ; ; note.half_up:L3;288.0,137.3,341.3,266.3
            image_id, _, _, _, _, full_category, bb = line.split(';')
            category, position = full_category.split(':')
            x1, y1, x2, y2 = bb.split(',')         

            if not image_id in images:
                images[image_id] = cv2.imread(image_folder + '/' + image_id, True)
                
            image_width = images[image_id].shape[1]
            image_height = images[image_id].shape[0]

            # Shape
            left = int(float(x1))
            top = int(float(y1))
            right = int(float(x2))
            bottom = int(float(y2))   
            
            X_shape.append(images[image_id][top:bottom,left:right])
            Y_shape.append(category)
            
            # Position        
            center_x = left + (right - left) / 2
            center_y = top + (bottom - top) / 2
            
            pos_left = int( max(0, center_x - fixed_width_position / 2) )
            pos_right = int( min(image_width, center_x + fixed_width_position / 2) )
            pos_top = int( max(0, center_y - fixed_height_position / 2) )
            pos_bottom = int( min(image_height, center_y + fixed_height_position / 2) )
    
            pad_left = int( abs( min(0, center_x - fixed_width_position / 2) ) )
            pad_right = int( abs( min(0, image_width - (center_x + fixed_width_position / 2)) ) )
            pad_top = int( abs( min(0, center_y - fixed_height_position / 2) ) )
            pad_bottom = int( abs( min(0, image_height - (center_y + fixed_height_position / 2)) ) )
            
            image_position = images[image_id][pos_top:pos_bottom,pos_left:pos_right]
            image_position = np.stack(
                            [np.pad(image_position[:,:,c],
                                   [(pad_top, pad_bottom), (pad_left, pad_right)],
                                   mode='symmetric')
                             for c in range(3)], axis=2)
            
            X_position.append(image_position)
            Y_position.append(position)

    return X_shape,Y_shape, X_position, Y_position


def get_categorical_map_from_list(Y):
    categories = set(Y)
    category_map = dict([(char,i) for i, char in enumerate(categories)])
    return category_map
    
def get_model(height, width, channels, categories, base = 'vgg16'):
    input_tensor = Input(name='input', shape=(height, width, channels))

    if base == 'vgg16':
        pretrained = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    if base == 'resnet':
        pretrained = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
        
    for layer in pretrained.layers:
        layer.trainable = True

    x = pretrained.output
    x = Conv2D(categories, kernel_size=(1, 1), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='output_class')(x)
    model = Model(inputs=pretrained.inputs, outputs=x)

    return model


if __name__ == "__main__":
    ### Settings
    img_height = 40
    img_width = 40
    
    img_pos_width = 112
    img_pos_height  = 224
    K.set_image_data_format("channels_last")
    ###

    parser = argparse.ArgumentParser(description='Mensural symbol classification (trainer).')
    parser.add_argument('-corpus', dest='train_set', type=str, required=True, help='Path to the corpus file.')
    parser.add_argument('-images', dest='corpus', type=str, required=True, help='Path to the images.')
    #parser.add_argument('-export_vocabulary', dest='vocabulary_map_file', type=str, required=True, help='Path to export the vocabulary map (integers <-> categories)')
    args = parser.parse_args()

    # Data preparation
    X_shape, Y_shape, X_position, Y_position = load_data(args.corpus, args.train_set)
    category_map = get_categorical_map_from_list(Y_shape)
    num_shapes = len(category_map)
    position_map = get_categorical_map_from_list(Y_position)
    num_positions = len(position_map)

    # Training shape
    X_shape = [cv2.resize(image, (img_width, img_height)) for image in X_shape]
    X_shape = np.asarray(X_shape).reshape(len(X_shape),img_height, img_width, 3)
    X_shape = (255. - X_shape) / 255.

    Y_shape = [category_map[c] for c in Y_shape]
    shape_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_shape),
                                                 Y_shape)
                                                 
    Y_shape = to_categorical(Y_shape, num_classes=num_shapes)
    Y_shape = np.asarray(Y_shape)

    model_shape = get_model(height = img_height, width = img_width, channels = 3, categories = num_shapes, base = 'vgg16')
    model_shape.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_shape.fit(X_shape, Y_shape, class_weight = shape_weights,
                    batch_size=16, epochs=30, verbose=2, validation_split=0.2,
                    callbacks=[ModelCheckpoint('shape_classifier.h5', monitor='val_acc', verbose=2, save_best_only=True)])    
    
    # Training position
    X_position = [cv2.resize(image, (img_pos_width, img_pos_height)) for image in X_position]
    X_position = np.asarray(X_position).reshape(len(X_position), img_pos_height, img_pos_width, 3)
    X_position = (255. - X_position) / 255.

    Y_position = [position_map[c] for c in Y_position]
    position_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_position),
                                                 Y_position)
    Y_position = to_categorical(Y_position, num_classes=num_positions)
    Y_position = np.asarray(Y_position)
    
    model_position = get_model(height = img_pos_height, width = img_pos_width, channels = 3, categories = num_positions, base = 'vgg16')
    model_position.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_position.fit(X_position, Y_position, class_weight = position_weights, 
                        batch_size=16, epochs=30, verbose=2, validation_split = 0.2, 
                        callbacks=[ModelCheckpoint('position_classifier.h5', monitor='val_acc', verbose=2, save_best_only=True)])
