# Traffic Light Detection AI
This project uses PyTorch to train an AI model capable of determining whether a given image contains a traffic light. It is trained on ~20,000 images from LISA Traffic Light Dataset. You can use my trained model to test images for the presence of traffic lights, or you can alter the code to train a model based on your own images.

## Technologies Used
- Python 3.12
- [PyTorch](https://pytorch.org/)
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

My model was over 99% accurate when tested on images from LISA. However, I expect this accuracy to be less when tested on images from other parts of the world, as LISA is gathered from California exclusively.

### Training Your Own Model
- Organize data into traffic_light and no_traffic_light directories. (`LISA_organize_script.py` organizes the dayTrain and nightTrain images into traffic_light and no_traffic_light. Adjust appropriate directories if you use this script).
- Adjust appropriate directories within `train.py`, `test.py` to point to appropriate directories on your machine.
- Run `train.py` and, optionally, `test.py` to train and test your model. It will replace the model located at models/traffic_light_model.pth<br/>

Note that the Flask app will now use your trained model rather than mine. Also, I assume this program could be used to train any Binary Classification model, not just traffic lights.

## Applications
Programs like this one are the reason CAPTCHA tests have gotten more complicated and difficult in recent years. A test such as this one

can be easily solved through machine learning. This has led to new types of CAPTCHA tests, such as these:

CAPTCHA tests must continue to improve to ensure security against AI powered bots.
