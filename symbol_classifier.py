import numpy as np
from keras.models import load_model
from keras import backend as K
import logging


__all__ = [ 'SymbolClassifier' ]


class SymbolClassifier:

    logger = logging.getLogger('SymbolClassifier')


    def __init__(self, model_shape_path, model_position_path, vocabulary_shape, vocabulary_position):
        self.logger.info('Loading models...')

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

        self.logger.info('Models loaded')


    def predict(self, shape_image, position_image, n):
        # Predictions
        shape_prediction_all = self.model_shape.predict(shape_image)
        #self.logger.info(shape_prediction_all)
        #shape_prediction = np.argmax(shape_prediction_all)
        #self.logger.info(shape_prediction)
        shape_prediction = np.flip(np.argsort(shape_prediction_all.flatten()))[0:n] # Equivalent to argmax returning the index of the n maxmimum values
        #self.logger.info(shape_prediction)

        position_prediction_all = self.model_position.predict(position_image)
        #self.logger.info(position_prediction_all)
        #position_prediction = np.argmax(position_prediction_all)
        #self.logger.info(position_prediction)
        position_prediction = np.flip(np.argsort(position_prediction_all.flatten()))[0:n]
        #self.logger.info(position_prediction)

        return ([self.shape_vocabulary[x] for x in shape_prediction], [self.position_vocabulary[x] for x in position_prediction])
