import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K


__all__ = [ 'SymbolClassifier' ]


class SymbolClassifier:
    ### Settings
    img_height = 40
    img_width = 40

    img_pos_width = 112
    img_pos_height = 224

    page = None


    def __init__(self, model_shape_path, model_position_path, vocabulary_shape, vocabulary_position):
        print("[classifiers.SymbolClassifier] Loading models...")

        K.set_image_data_format("channels_last")
        if K.backend() == 'tensorflow':
            import tensorflow as tf    # Memory control with Tensorflow
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)

        self.model_shape = load_model(model_shape_path)
        self.model_shape._make_predict_function() # Workaround to solve multithreading issues with flask: https://github.com/keras-team/keras/issues/2397#issuecomment-306687500
        self.model_position = load_model(model_position_path)
        self.model_position._make_predict_function() # Workaround to solve multithreading issues with flask: https://github.com/keras-team/keras/issues/2397#issuecomment-306687500

        shape_vocabulary = np.load(vocabulary_shape).item()  # Category -> int
        self.shape_vocabulary = dict((v, k) for k, v in shape_vocabulary.items())  # int -> Category

        position_vocabulary = np.load(vocabulary_position).item()  # Category -> int
        self.position_vocabulary = dict((v, k) for k, v in position_vocabulary.items())  # int -> Category

        print("[classifiers.SymbolClassifier] Models loaded")


    def set_page(self, page):
        self.page = page
    

    def unset_page(self):
        self.page = None


    def has_page(self):
        return not self.page is None


    def crop_page(self, left, top, right, bottom):
        if not self.has_page:
            return None, None

        # Shape
        shape_image = self.page[top:bottom, left:right]
	    #drizo
        #cv2.imwrite('debug_shape.png', shape_image)
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

        #cv2.imwrite('debug_position.png', position_image)

        position_image = np.asarray(position_image).reshape(1, self.img_pos_height, self.img_pos_width, 3)
        position_image = (255. - position_image) / 255.

        return (shape_image, position_image)


    def predict(self, left, top, right, bottom):
        if not self.has_page():
            return None, None

        #print("[classifiers.SymbolClassifier] predicting bbox({}, {}, {}, {})".format(left, top, right, bottom))

        shape_image, position_image = self.crop_page(left, top, right, bottom)

        # Predictions
        shape_prediction = self.model_shape.predict(shape_image)
        shape_prediction = np.argmax(shape_prediction)

        position_prediction = self.model_position.predict(position_image)
        position_prediction = np.argmax(position_prediction)

        return (self.shape_vocabulary[shape_prediction], self.position_vocabulary[position_prediction])
