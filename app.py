from flask import Flask,request, jsonify, Response
from flask_restful import Resource, reqparse, Api
from flask_cors import CORS

from alexnet_detection import alexnet_predict
from resnet50_detection import resnet_predict
from mobilenet_detection import mobilenet_predict
from densenet_inference import densenet_predict

#Instantiate a flask object
app = Flask(__name__)
CORS(app)

#Instantiate Api object
api = Api(app)


@app.route("/")
def index():
    return "Homepage of Ear Image Classification"


@app.route('/classify_alexnet', methods=['POST'])
def classify_alexnet():
  # Check if an image file was uploaded
  if 'image' not in request.files:
    return jsonify({'error': 'No image file found'})

  img_path = request.files['image'].stream

  # Perform image classification with AlexNet model
  predicted_class, predicted_prob = alexnet_predict(img_path)

  # Return the prediction result as JSON response
  return jsonify({'class': predicted_class, 'prob': predicted_prob})


@app.route('/classify_resnet', methods=['POST'])
def classify_resnet():
  # Check if an image file was uploaded
  if 'image' not in request.files:
    return jsonify({'error': 'No image file found'})

  img_path = request.files['image'].stream

  # Perform image classification with ResNet model
  predicted_class, predicted_prob = resnet_predict(img_path)

  # Return the prediction result as JSON response
  return jsonify({'class': predicted_class, 'prob': predicted_prob})


@app.route('/classify_mobilenet', methods=['POST'])
def classify_mobilenet():
  # Check if an image file was uploaded
  if 'image' not in request.files:
    return jsonify({'error': 'No image file found'})

  img_path = request.files['image'].stream

  # Perform image classification with MobileNet model
  predicted_class, predicted_prob = mobilenet_predict(img_path)

  # Return the prediction result as JSON response
  return jsonify({'class': predicted_class, 'prob': predicted_prob})


@app.route('/classify_densenet', methods=['POST'])
def classify_densenet():
  # Check if an image file was uploaded
  if 'image' not in request.files:
    return jsonify({'error': 'No image file found'})

  img_path = request.files['image'].stream

  # Perform image classification with MobileNet model
  predicted_class, predicted_prob = densenet_predict(img_path)

  # Return the prediction result as JSON response
  return jsonify({'class': predicted_class, 'prob': predicted_prob})


if __name__== '__main__':
  app.run(debug=True)
