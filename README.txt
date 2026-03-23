File Structure

venv/
.gitignore
models/
    simple_CNN.py
    Add more models here for testing
data/ - folder holding all images in there class folder
    0/ - folder conatains all images for that class
        img1.jpg
        img2.jpg - images 
        ...
    1/
    2/
    3/
    4/
    5/
    6/
    7/
    8/
    9/
Camera_windows.py - for developing or just running on Camera_windows
Camera.py - for running on the jetson nano 

%% Rest run on either the Jetson nano (Linux - Ubuntu) or windows - MacOS is untested

Dataloader.py
dataset.py
train.py
trainning.py
visionPreprocess.py


