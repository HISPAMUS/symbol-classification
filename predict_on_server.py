import argparse
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K

class Classifier:
    ### Settings
    img_height = 40
    img_width = 40

    img_pos_width = 112
    img_pos_height = 224

    def __init__(self, model_shape_path, model_position_path, vocabulary_shape, vocabulary_position):
        K.set_image_data_format("channels_last")

        self.model_shape = load_model(model_shape_path)
        self.model_position = load_model(model_position_path)

        shape_vocabulary = np.load(vocabulary_shape).item()  # Category -> int
        self.shape_vocabulary = dict((v, k) for k, v in shape_vocabulary.items())  # int -> Category

        position_vocabulary = np.load(vocabulary_position).item()  # Category -> int
        self.position_vocabulary = dict((v, k) for k, v in position_vocabulary.items())  # int -> Category

    def set_page(self, page):
        self.page = page

    def predict(self, left, top, right, bottom):
        # Shape
        shape_image = self.page[top:bottom, left:right]
        shape_image = [cv2.resize(shape_image, (self.img_width, self.img_height))]
        shape_image = np.asarray(shape_image).reshape(1, self.img_height, self.img_width, 3)
        shape_image = (255. - shape_image) / 255.

        # Position [mirror effect for boxes close to the limits]
        image_height, image_width, _  = self.page.shape

        center_x = left + (right - left) / 2
        center_y = top + (bottom - top) / 2

        pos_left = int(max(0, center_x - self.img_pos_width / 2))
        pos_right = int(min(image_width, center_x + self.img_pos_width / 2))
        pos_top = int(max(0, center_y - self.img_pos_height / 2))
        pos_bottom = int(min(image_height, center_y + self.img_pos_height / 2))

        pad_left = int(abs(min(0, center_x - self.img_pos_width / 2)))
        pad_right = int(abs(min(0, image_width - (center_x + self.img_pos_width / 2))))
        pad_top = int(abs(min(0, center_y - self.img_pos_height / 2)))
        pad_bottom = int(abs(min(0, image_height - (center_y + self.img_pos_height / 2))))

        position_image = self.page[pos_top:pos_bottom, pos_left:pos_right]
        position_image = np.stack(
            [np.pad(position_image[:, :, c],
                    [(pad_top, pad_bottom), (pad_left, pad_right)],
                    mode='symmetric')
             for c in range(3)], axis=2)

        position_image = np.asarray(position_image).reshape(1, self.img_pos_height, self.img_pos_width, 3)
        position_image = (255. - position_image) / 255.

        # Predictions
        shape_prediction = self.model_shape.predict(shape_image)
        shape_prediction = np.argmax(shape_prediction)

        position_prediction = self.model_position.predict(position_image)
        position_prediction = np.argmax(position_prediction)

        return self.shape_vocabulary[shape_prediction] + ':' + self.position_vocabulary[position_prediction]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mensural symbol classification (predictor on the server).')
    parser.add_argument('-image',    dest='image_shape', type=str, required=True)
    parser.add_argument('-bounding_box', dest='bounding_box', nargs=4, type=int, required=True, help="Bounding box in the form of [left, top, right, bottom]")
    parser.add_argument('-model_shape',    dest='model_shape', type=str, default='model/shape_classifier.h5')
    parser.add_argument('-model_position', dest='model_position',    type=str, default='model/position_classifier.h5')
    parser.add_argument('-vocabulary_shape',    dest='vocabulary_shape', type=str, default='model/category_map.npy')
    parser.add_argument('-vocabulary_position', dest='vocabulary_position',    type=str, default='model/position_map.npy')
    args = parser.parse_args()

    # Create classifier, which loads the models and the dictionary for the vocabularies
    clf = Classifier(args.model_shape, args.model_position, args.vocabulary_shape, args.vocabulary_position)

    # Load page
    page = cv2.imread(args.image_shape)
    clf.set_page(page)

    # Classify bounding box
    prediction = clf.predict(args.bounding_box[0], args.bounding_box[1], args.bounding_box[2], args.bounding_box[3])
    print(prediction)



