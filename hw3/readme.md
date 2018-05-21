# Homework #3: Chord Estimation
Chord is a key component of a song which represent the structure and the flow of a song. In music making and song cover it is important to estimate chords.  
As we shared [Chordify](https://chordify.net) from the lecture there are demand for estimating chord structure. In this homework you are going to build your own neural network model to estimate chord from given chroma or beat-synchronous chroma.

## Dataset
We use the [RWC](https://staff.aist.go.jp/m.goto/RWC-MDB/) for this homework. The RWC dataset is the large-scale music dataset which contains various music database with MIDI, lyrics. In this homework we will use Popular Music Database with 100 songs of Japanese pop music and it contains chord annotation on [RWC Chord Annotation](https://github.com/tmc323/Chord-Annotations).  
Because RWC dataset consists of commercial songs it is illegal to share the audio files. Instead of audio file we offer chromagram, chord annotation and beat information in numpy format. The dataset is contained in this repository and you don't need any special installation. The information of each feature is like below.  

* Chroma : chroma energy normalized(cens) chroma with 44100 sampling rate, 512 hop size.
* Chord : 0 is silence without any chord, 1 to 12 are major chords and 13 to 24 are minor chords. Chord is annotated in frame level.
* Beat : Beat array contains the number of frame which is beat.

## Baseline Code
 You should download dataset from [KLMS](http://klms.kaist.ac.kr/mod/ubfile/view.php?id=249693) and once you download both dataset and code, file structure is like below. In the dataset folder there are chroma, chord and beat information. Baseline code classify chords using the dataset with 1 layer of bidirectional LSTM. Detailed explanation of each file is in the comment so please check in each file.

* hw3
  * dataset
    * chroma
    * chord
    * beat
  * data_manager.py
  * model_archive.py
  * model_wrapper.py
  * main_frame.py
  * main_beatsync.py
  * result_visualizer.py

Because feature is offered in this homework you don't need any feature extraction. Simply run 'main_frame.py' or 'main_beatsync.py' and then training will start. If the training runs successfully it will display losses and accuracies in a batch. If training is finished the 'export' folder will automatically generated and inside the export folder model, chromagram, annotation and prediction will be saved.

```
Data Loaded
Train Ratio : 57.0%, Test Ratio : 19.0%, Valid Ratio : 22.0%

--------- Training Start ---------
Learning Rate : 0.0001
Epoch [001/500] acc : 0.0887 - val_acc : 0.1262 | loss : 3.1647 - val_loss : 3.079
Learning Rate : 0.0001
Epoch [002/500] acc : 0.1045 - val_acc : 0.1264 | loss : 3.0296 - val_loss : 2.9681
Learning Rate : 0.0001
Epoch [003/500] acc : 0.1052 - val_acc : 0.1262 | loss : 2.9766 - val_loss : 2.9315
Learning Rate : 0.0001
Epoch [004/500] acc : 0.1201 - val_acc : 0.1359 | loss : 2.9404 - val_loss : 2.8892
...
-------- Training Finished -------

Test Accuracy :
Exported files to /home/cjb3549/Projects/GCT634/hw3/export/
```

After you run one of the main code then run 'result_visualizer.py'. This code visualize chromagram, annotation and prediction based on the trained result. Please check IMPORT_DIR before you run the code.

## Improving Algorithms
In this homework you should improve the model to increase the performance of chord estimation. Here is some idea to improve the model. First you can simply adjust parameters of neural network or change the structure of RNN. Because RNN trains parameter from memories the batch size is also an important parameter.  
In recent researches there is a neural network structure called CRNN which combines CNN and RNN. In CRNN, RNN follows multilayer of the CNN so that RNN can train the complex structure of a feature. Also you can try HMM from the lecture. But it might need a lot of effort you should make your own 'model_wrapper.py' and change the 'Train' part of main.  
Chromagram is fixed but you can try different level of beat. You can divide beat and train model. Then you should convert prediction into original beat level and compare the result.

* Adjusting parameters
  * Learning rate
  * Patience value
  * Batch size
  * Optimizers

* RNN
  * Number of layers
  * Number of RNN hidden units
  * Type of RNN cell
  * Unidirectional vs bidirectional

* CNN + RNN
  * Network with combination of CNN and RNN
  * Number of CNN layers
  * CNN structure

* HMM
  * HMM using scikit learn library

## Deliverables
You should submit your python code and homework report in pdf to KLMS.
The report should include:
* Algorithm Description
* Experiments and Results
* Discussion
