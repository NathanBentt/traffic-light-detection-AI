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

## Applications
AI models like this one are the reason CAPTCHA tests have gotten more complicated and difficult in recent years. While my model and program cannot directly solve CAPTCHA tests, similar ideas could be applied to create a bot capable of solving CAPTCHAs. A test such as this one:
![simple traffic light captcha](https://github.com/user-attachments/assets/1a656162-f0c0-4dda-871c-3174db88aa92)<br/>
can be easily solved through machine learning. This has led to new types of CAPTCHA tests, such as text-based ones like this:<br/>
![word captcha](https://github.com/user-attachments/assets/a23ec365-7183-4931-8028-d9f65612cb4b)<br/>
and even some newer, 3D model ones like this:<br/>
![3d captcha](https://github.com/user-attachments/assets/d0db52ca-4406-4595-a509-f6fbc172dc3e)<br/>
CAPTCHA tests must continue to improve to ensure security against AI powered bots.
