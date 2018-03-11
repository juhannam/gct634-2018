# Homework #0: Getting Familiar With Librosa

The purpose of this homework is getting you familiar with [Librosa](https://librosa.github.io/), the Python package for music and audio analysis. We are going to heavily use the library for audio representations and feature extraction throughout this course.  


## Installation 
First, make sure that you have a Python programming environment on your machine. If you are using Windows, you should go to https://www.python.org/ and install Python. I recommend you to use Python 2 (version 2.7) to avoid possible dependency problems. If you are using Mac OS X, I strongly recommend you to install [Virutualenv](https://virtualenv.pypa.io/en/stable/) and use Python in the virtual environment (see below) because the system-supported installation on MAC OS X has a trouble with Librosa (to my experience).  Once you are ready to use Python, go to the Librosa website (https://librosa.github.io/) and install it. 

>#### Virtualenv  (tested on MAC OS X)
>
>To create a virtual environment: 
>```sh
>$ mkdir gct634
>$ virtualenv ./gct634
>```
>To activate the created environment (the **bin** folder is under the **gct634** folder): 
>```sh
>$ cd gct634
>$ source ./bin/activate
>```
>Then, your terminal prompt will change to include **(gct634)** and now you are in the virtual environment.
>
>In the virtual environment, install Python packages:
>```sh
>$ pip install librosa
>```

You also need to install [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) for matrix computation and visualization. Before installing them, make sure you already have them in your Python setup.  A nice tutorial of Numpy is linked at 
http://cs231n.github.io/python-numpy-tutorial/.  


## Audio Representations
Your first mission is loading audio files and displaying it  as 2D time-frequency representations.  The tutorial code is provided in the following link.

[GCT634 (2018) Audio Representations.ipynb](https://drive.google.com/file/d/1ZqB1u5YAVLWVLyRGdeHeM40RVahN9FLJ/view?usp=sharing)

Note that it is a cloud-based Jupiter notebook, **Google Colab** that I mentioned in the class.  You can run each section of the code one by one by pressing the "run cell" buttons on the web page. If you want to run the code on your local Python environment, skip the first section ( 1. Install Librosa and FFmpeg ), copy each of the section to your editor and run it. 


## Analyzing Your Own Music Files
Once you figure out how to display the audio representations as an image, bring your own music files and compare what you hear to what you see.  If you cannot find any mp3 source, you can get some here in the following links. 

* [Free Music Archive](http://freemusicarchive.org/)
* [Bensound]( https://www.bensound.com)
* [CCMixter](http://ccmixter.org/view/media/home)


## Studying Further
The Librosa webpage has the [tutorial](http://librosa.github.io/librosa/tutorial.html) section that covers various functions of audio feature extraction and music analysis. If you are interested, you can further analyze your music file using them.  

Another great tutorial is [Steve Tjoa's CCRMA MIR Workshop Notes](https://musicinformationretrieval.com/). It is not only nicely orgarnized in Jupiter notebook format but also greatly covers the topics that we handle in our course. 

