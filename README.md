# Human-Detection
human detection model for flood disaster response

Download the dataset 
 - https://www.kaggle.com/datasets/rgbnihal/c2a-dataset/data

make changes in the folders and save the data folder in code files
 -  data/                           
    │   ├── All labels with Pose information/
    |   |   ├── labels/
    │   │   ├── *.txt 
    │   │
    │   ├── train/
    │   │   ├── images/  
    │   │   ├── labels/   
    │   │   ├── annotations.json 
    │   │
    │   ├── val/
    │   │   ├── images/      
    │   │   ├── labels/
    │   │   ├── annotations.json
    │   │
    │   ├── test/
    │       ├── images/       
    │       ├── labels/
    │       ├── annotations.json

make virtual environment
 - pip install -r requirements.txt

train the model
 - python train_model.py

test the model locally
 - python detect_webcam.py

run the web page (upload image and find human)
 - cd web_app
 - python app.py

# FIRST WORKING