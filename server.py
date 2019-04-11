import argparse
import flask
from flask import request, jsonify
import os
import logging
from symbol_classifier import SymbolClassifier
from image_storage import ImageStorage


classifier = None
storage = None
app = flask.Flask(__name__)
logger = logging.getLogger('server')


def message(text):
    return jsonify({ 'message': text })


@app.route('/image', methods=['POST'])
def image_save():
    if request.files.get('image') and request.form.get('id'):
        id = request.form['id']
        request.files['image'].save(storage.path(id))
        return message(f'Image [{id}] stored'), 200
    else:
        return message('Missing data'), 400


@app.route('/image/<id>', methods=['GET'])
def image_check(id):
    if storage.exists(id):
        return message(f'Image [{id}] exists'), 200
    else:
        return message(f'Image [{id}] does not exist'), 404


@app.route('/image/<id>', methods=['DELETE'])
def image_delete(id):
    if storage.exists(id):
        os.remove(storage.path(id))
        return message(f'Image [{id}] deleted'), 200
    else:
        return message(f'Image [{id}] does not exist'), 404


@app.route('/image/<id>/bbox', methods=['POST'])
def predict(id):
    if not storage.exists(id):
        return message(f'Image [{id}] does not exist'), 404
    
    try:
        left = int(request.form['left'])
        top = int(request.form['top'])
        right = int(request.form['right'])
        bottom = int(request.form['bottom'])
        n = int(request.form.get('predictions', "1"))
    except ValueError as e:
        return message('Wrong input values'), 400

    try:
        shape_image, position_image = storage.crop(id, left, top, right, bottom)
    except Exception as e:
        return message('Error cropping image'), 400

    shape, position = classifier.predict(shape_image, position_image, n)
    if shape is None or position is None:
        return message('Error predicting symbol'), 404
    
    result = { 'shape': shape, 'position': position }
    return jsonify(result), 200


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Mensural symbol classification (predictor on the server).')
    parser.add_argument('-model_shape',    dest='model_shape', type=str, default='model/shape_classifier.h5')
    parser.add_argument('-model_position', dest='model_position',    type=str, default='model/position_classifier.h5')
    parser.add_argument('-vocabulary_shape',    dest='vocabulary_shape', type=str, default='model/shape_map.npy')
    parser.add_argument('-vocabulary_position', dest='vocabulary_position',    type=str, default='model/position_map.npy')
    parser.add_argument('-port', dest='port', type=int, default=8888)
    parser.add_argument('-image_storage', dest='image_storage', type=str, default='images')
    args = parser.parse_args()

    # Initialize image storage
    storage = ImageStorage(args.image_storage)

    # Create classifier, which loads the models and the dictionary for the vocabularies
    classifier = SymbolClassifier(args.model_shape, args.model_position, args.vocabulary_shape, args.vocabulary_position)

    # Start server, 0.0.0.0 allows connections from other computers
    app.run(host='0.0.0.0', port=args.port)
