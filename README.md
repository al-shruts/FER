# Face expression recognition

## FER from camera video

### Installation
To work with facial expression recognition from camera video, you need to install the following dependencies

```sh
# If You Windows 10
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

```sh
# Anyway  
pip install tensorflow numpy face_recognition opencv-python
```

### Launch
To run you need a working camera on your computer

```sh
python main.py
```

## FER Learning

### Installation
To start training the model, you need to install the following dependencies
```sh
pip install tensorflow numpy pandas keras-tuner livelossplot git+https://github.com/yaledhlab/vggface.git keras_vggface keras_applications
```
* But you also need data

### Launch
To run you need to run 
```sh 
FER.py
```
or (if you want to find the optimal parameters)
```sh 
tuner.py
```
