import argparse
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K

if __name__ == "__main__":
    ### Settings
    img_height = 40
    img_width = 40

    img_pos_width = 112
    img_pos_height = 224
    K.set_image_data_format("channels_last")

    parser = argparse.ArgumentParser(description='Mensural symbol classification (predictor).')
    parser.add_argument('-image_shape',    dest='image_shape', type=str, required=True)
    parser.add_argument('-image_position', dest='image_position',    type=str, required=True)
    parser.add_argument('-model_shape',    dest='model_shape', type=str, default='model/shape_classifier.h5')
    parser.add_argument('-model_position', dest='model_position',    type=str, default='model/position_classifier.h5')
    parser.add_argument('-vocabulary_shape',    dest='vocabulary_shape', type=str, default='model/category_map.npy')
    parser.add_argument('-vocabulary_position', dest='vocabulary_position',    type=str, default='model/position_map.npy')
    args = parser.parse_args()

    # SHAPE
    image_shape = cv2.imread(args.image_shape)
    image_shape = cv2.resize(image_shape, (img_width, img_height))
    image_shape = np.asarray(image_shape).reshape(1, img_height, img_width, 3)
    image_shape = (255. - image_shape) / 255.

    model_shape = load_model(args.model_shape)
    shape_prediction = model_shape.predict(image_shape)
    shape_prediction = np.argmax(shape_prediction)

    shape_vocabulary = np.load(args.vocabulary_shape).item()     # Category -> int
    shape_vocabulary = dict((v, k) for k, v in shape_vocabulary.items())     # int -> Category

    # VOCABULARY
    image_position = cv2.imread(args.image_position)
    image_position = cv2.resize(image_position, (img_pos_width, img_pos_height))
    image_position = np.asarray(image_position).reshape(1, img_pos_height, img_pos_width, 3)
    image_position = (255. - image_position) / 255.

    model_position = load_model(args.model_position)
    position_prediction = model_position.predict(image_position)
    position_prediction = np.argmax(position_prediction)

    position_vocabulary = np.load(args.vocabulary_position).item()     # Category -> int
    position_vocabulary = dict((v, k) for k, v in position_vocabulary.items())     # int -> Category

    print(shape_vocabulary[shape_prediction]+':'+position_vocabulary[position_prediction])

