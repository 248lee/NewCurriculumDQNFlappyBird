# Curriculum DQN flappy bird with tensorflow2

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

## Human playing
Run ```python humanPlay.py``` to play the flappy bird game by yourself. Press SPACE to jump, 0 to fire.
