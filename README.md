# Ear Image Classification Project 

## Introduction 
This project implements a flask app to detect whether the ear is normal or ill in image. 

We use three different classification models: AlexNet, ResMet, and MobileNet.

The user could upload a single image in our webpage. Click "classify-<ModelName>" button and get the classification result with a specific confidence level of corresponding model.


## Prerequisite
Need to have Python3, PyTorch library and Google Chrome


## How to run it
Run the following command to start the backend application

```
python3 app.py
```


Run the following command to start a http server for the webpage:

```
python -m http.server
```

The following command would host the webpage in localhost:8000.

Open the webpage in Chrome to test with: 
http://localhost:8000/



## Project Structure
1. app.py: Use flask framework to build the application. 

2. alexnet_detection.py: Do inference with AlexNet model.

3. resnet50_detection.py: Do inference with ResNet model.

4. prelim_mobilenet_detection.py: Do inference with MobileNet model.

5. AlexNet_model.pth: Saved AlexNet model

6. ResNet_model.pth: Saved ResNet model

7. MobileNet_model.pth: Saved MobileNet model

8. index.html: Implement a webpage


