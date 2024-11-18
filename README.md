# Traffic Light Detection AI
This project uses [PyTorch](https://pytorch.org/) to train an AI model capable of determining whether a given image contains a traffic light. It is trained on ~20,000 images from [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset). I used [ResNet18](https://pytorch.org/vision/main/models/resnet.html) as a foundation, replacing and training the top layer for traffic light detection. You can use my trained model to test images for the presence of traffic lights, or you can make some changes and train a model based on your own images.

## Technologies Used
- Python 3.12
- PyTorch
- Pandas
- Pillow
- ResNet18
- Flask
- HTML
- CSS

## Requirements
`pip install flask torch torchvision pillow pandas`
- Flask, torch, torchvision, pillow used in Flask web application
- torch and torchvision used in creating, training, and testing model
- pandas used to process images from LISA

## Usage
### Using My Model
- Run `app.py` to launch the Flask web app
- Upload and analyze an image - the model will predict if that image contains a traffic light.<br/>

My model was over 99% accurate when tested on images from LISA. However, I expect this accuracy to be less when tested on images from other parts of the world, as LISA is gathered from San Diego, California exclusively.

### Training Your Own Model
- Organize data into traffic_light and no_traffic_light directories. (`LISA_organize_script.py` organizes the dayTrain and nightTrain images into traffic_light and no_traffic_light. Adjust appropriate directories if you use this script).
- Adjust appropriate directories within `train.py`, `test.py` to point to appropriate directories on your machine.
- Run `train.py` and, optionally, `test.py` to train and test your model. It will replace the model located at models/traffic_light_model.pth<br/>

Note that the Flask app will now use your trained model rather than mine. Also, I assume this program could be used to train any Binary Classification model, not just traffic lights.
