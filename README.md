# Curriculum DQN flappy bird with tensorflow2
<img src="https://i.imgur.com/MFnCmpD.gif" />

### This project is only tested in the conda environment with python=3.10.13, Ubuntu18.04 and 20.04.
# Environment Installation:
1. Download Miniconda or Anaconda from the official website.
2. Follow the instructions to install conda.
3. ```conda create --name [any-name-you-want] python=3.10.13```
4. ```conda activate [your-environment-name]```


# Dependencies Installation:
1. ```pip install tensorflow[and-cuda] # [and-cuda] is needed for GPU training```
2. ```pip install pygame```
3. ```pip install matplotlib```
4. ```pip install opencv-python```
5. If you prefer GUI training panel, also install:
```conda install anaconda::pyqt```

# How to Run?
## Training:  
Run ```python app.py``` gives you a GUI panel to config the training process, and you can select a training stage with it.
If your system doesn't support GUI (eg. colab, WSL), run ```python non_gui.py``` to train. Modify the parameters in non_gui.py to config the training process.

### Train Old or New Model
This project supports resuming the training process of the current model after the training process was terminated. When you start to run the training program (either through app.py or non_gui.py), the program automatically chooses to train a new model or resume the existing one, depending on whether the file ```model/FlappyBird.h5``` is existing.

There is a zip file in the ```model``` directory that contains the model I've trained. Unzip it and execute the training process (with stage=2 and num_of_actions=3) to watch my model playing the game!!!

### Train New Model
Before training a new model, run ```python non_gui_train_new.py``` before running non_gui.py (or press the "Train New Network" button for the GUI users). This will delete all the log files during training, as well as the model ```model/FlappyBird.h5``` so that the process will create a new network to train. Remember that when you train a new model, the current model will be deleted, so make sure that **YOU'VE MADE A COPY**!!

## Human playing
Run ```python humanPlay.py``` to play the flappy bird game by yourself. Press SPACE to jump, 0 to fire.
