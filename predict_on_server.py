import argparse
import cv2
import numpy as np
from symbol_classifier import SymbolClassifier
import flask
import tempfile
import os


classifier = None
app = flask.Flask(__name__)


def load_page_stream(image):
    classifier.set_page(
        cv2.imdecode(
            np.asarray(
                bytearray(image.read()),
                dtype=np.uint8
            ),
            -1 # No color transformation, load image "as is"
        )
    )


def load_page(image):
    page = cv2.imread(image)
    classifier.set_page(page)


@app.route("/page", methods=["GET", "POST", "DELETE"])
def set_page():
    if flask.request.method == "GET":
        if classifier.has_page():
            return flask.jsonify({ "success": True }), 200
        else:
            return flask.jsonify({ "success": False }), 404
    elif flask.request.method == "POST":
        if flask.request.files.get("image"):
            #flask.request.files["image"].save(flask.request.files["image"].filename)
            #load_page(flask.request.files["image"].filename)
            load_page_stream(flask.request.files["image"]) # files dict contains stream-like objects (FileStorage type)
            return flask.jsonify({ "success": True }), 200
    elif flask.request.method == "DELETE":
        classifier.unset_page()
        return flask.jsonify({ "success": True }), 200


@app.route("/page/prediction", methods=["POST"])
def predict():
    result = { "success": False }

    if flask.request.method == "POST":
        if not classifier.has_page():
            result["message"] = "Page has not been loaded"
            return flask.jsonify(result), 404
        
        try:
            left = int(flask.request.form['left'])
            top = int(flask.request.form['top'])
            right = int(flask.request.form['right'])
            bottom = int(flask.request.form['bottom'])
        except ValueError as e:
            result["message"] = "Wrong input values"
            return flask.jsonify(result), 400

        shape, position = classifier.predict(left, top, right, bottom)
        if shape is None or position is None:
            return flask.jsonify(result), 404
        
        result["success"] = True
        result["shape"] = shape
        result["position"] = position
        return flask.jsonify(result), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mensural symbol classification (predictor on the server).')
    parser.add_argument('-model_shape',    dest='model_shape', type=str, default='model/shape_classifier.h5')
    parser.add_argument('-model_position', dest='model_position',    type=str, default='model/position_classifier.h5')
    parser.add_argument('-vocabulary_shape',    dest='vocabulary_shape', type=str, default='model/shape_map.npy')
    parser.add_argument('-vocabulary_position', dest='vocabulary_position',    type=str, default='model/position_map.npy')
    parser.add_argument('-port', dest='port', type=int, default=8888)
    args = parser.parse_args()

    # Create classifier, which loads the models and the dictionary for the vocabularies
    classifier = SymbolClassifier(args.model_shape, args.model_position, args.vocabulary_shape, args.vocabulary_position)

    # Load page
    #load_page(args.image_shape)

    # Classify bounding box
    #prediction = classifier.predict(args.bounding_box[0], args.bounding_box[1], args.bounding_box[2], args.bounding_box[3])
    #print(prediction)

    # Start server, 0.0.0.0 allows connections from other computers
    app.run(host='0.0.0.0', port=args.port)
