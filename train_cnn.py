import cv2
import numpy as np
import argparse

from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, Conv2D
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import glob
import json
import os.path
import sys

fixed_width_position = 112
fixed_height_position = 224

def load_data(json_folder, image_folder):
    X_shape = []
    Y_shape = []
    X_position = []
    Y_position = []

    for sample in glob.glob("{}/*".format(image_folder)):
        print(sample)
        image = cv2.imread(sample,cv2.IMREAD_COLOR)
        if image is None:
            continue

        if os.path.isfile("{}/{}.json".format(json_folder,os.path.basename(sample))):
            with open("{}/{}.json".format(json_folder,os.path.basename(sample))) as json_file:
                annotation = json.load(json_file)

                image_width = image.shape[1]
                image_height = image.shape[0]

                for page in annotation['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:
                                for symbol in region["symbols"]:

                                    # Shape
                                    left = int(float(symbol["bounding_box"]["fromX"]))
                                    top = int(float(symbol["bounding_box"]["fromY"]))
                                    right = int(float(symbol["bounding_box"]["toX"]))
                                    bottom = int(float(symbol["bounding_box"]["toY"]))

                                    X_shape.append(image[top:bottom,left:right])
                                    Y_shape.append(symbol["agnostic_symbol_type"])

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

                                    image_position = image[pos_top:pos_bottom,pos_left:pos_right]
                                    image_position = np.stack(
                                                    [np.pad(image_position[:,:,c],
                                                            [(pad_top, pad_bottom), (pad_left, pad_right)],
                                                            mode='symmetric')
                                                        for c in range(3)], axis=2)

                                    X_position.append(image_position)
                                    Y_position.append(symbol["position_in_staff"])

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
    ###

    parser = argparse.ArgumentParser(description='Mensural symbol classification (trainer).')
    parser.add_argument('--json', dest='json_folder', type=str, required=True, help='Path to the JSONs folder.')
    parser.add_argument('--image', dest='image_folder', type=str, required=True, help='Path to the images folder.')
    args = parser.parse_args()

    # Data preparation
    X_shape, Y_shape, X_position, Y_position = load_data(args.json_folder, args.image_folder)
    category_map = get_categorical_map_from_list(Y_shape)
    num_shapes = len(category_map)
    position_map = get_categorical_map_from_list(Y_position)
    num_positions = len(position_map)

    # Export dictionaries
    with open("shape_dictionary.json", "w") as outfile:
        json.dump(category_map, outfile)

    with open("position_dictionary.json", "w") as outfile:
        json.dump(position_map, outfile)

    # Training shape
    X_shape = [cv2.resize(image, (img_width, img_height)) for image in X_shape]
    X_shape = np.asarray(X_shape).reshape(len(X_shape),img_height, img_width, 3)
    X_shape = (255. - X_shape) / 255.

    Y_shape = [category_map[c] for c in Y_shape]
    shape_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(Y_shape),
                                                 y=Y_shape)

    shape_weights = {i:k for i,k in enumerate(shape_weights)}
    shape_weights = None
    Y_shape = to_categorical(Y_shape, num_classes=num_shapes)
    Y_shape = np.asarray(Y_shape)


    print("Shape model with {} samples".format(X_shape.shape[0]))
    model_shape = get_model(height = img_height, width = img_width, channels = 3, categories = num_shapes, base = 'vgg16')
    model_shape.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_shape.fit(X_shape, Y_shape, class_weight = shape_weights,
                    batch_size=16, epochs=15, verbose=2, validation_split=0.2,
                    callbacks=[ModelCheckpoint('shape_classifier.h5', monitor='val_accuracy', verbose=2, save_best_only=True)])

    # Training position
    X_position = [cv2.resize(image, (img_pos_width, img_pos_height)) for image in X_position]
    X_position = np.asarray(X_position).reshape(len(X_position), img_pos_height, img_pos_width, 3)
    X_position = (255. - X_position) / 255.

    Y_position = [position_map[c] for c in Y_position]
    position_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(Y_position),
                                                 y=Y_position)
    Y_position = to_categorical(Y_position, num_classes=num_positions)
    Y_position = np.asarray(Y_position)

    position_weights = {i:k for i,k in enumerate(position_weights)}
    position_weights = None

    model_position = get_model(height = img_pos_height, width = img_pos_width, channels = 3, categories = num_positions, base = 'vgg16')
    model_position.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Position model with {} samples".format(X_position.shape[0]))
    model_position.fit(X_position, Y_position, class_weight = position_weights,
                        batch_size=16, epochs=100, verbose=1, validation_split = 0.2,
                        callbacks=[ModelCheckpoint('position_classifier.h5', monitor='val_accuracy', verbose=2, save_best_only=True)])
