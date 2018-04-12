# Homework #2: Genre Classification
Genre classification is an important task that can be used many music applications. Your mission is to build your own Convolutional Neural Network (CNN) model to classify audio files into different music genres. Specifically, the goals of this homework are as follows:

* Experiencing the whole pipeline of deep learning based system: data preparation, feature extraction, model training and evaluation
* Getting familiar with the CNN architectures for music classification tasks
* Using Pytorch in practice

## Dataset
We use the [GTZAN](http://marsyasweb.appspot.com/download/data_sets/) dataset which has been the most widely used in the music genre classification task. The dataset contains 30-second audio files with the dataset split of 443 for training, 197 for validation and 290 for testing. Total 10 genres are the ones to classify, which include reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. You can download the dataset from [this link](https://drive.google.com/open?id=12sXcLOIylUZeSoigajSKAJQ1aWGtnxAv). 

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

```
$ cd gtzan
$ ls 
blues disco metal ... rock train_filtered.txt valid_filtered.txt test_filtered.txt
$ cd ..      # go back to your home folder for next steps
```

## Baseline Code
The source code of the baseline algorithm is provided so that you can easily start with the homework and also compare your own algorithm to it in performance. The baseline model extracts mel-spectrogram and has a simple setup of CNN model that includes convolutional layer, batch normalization, non-linearity function, maxpooling and dense layer. 


From the homework git repository, download the following Python files. 

* feature_extraction.py: loads audio files, extracts mel-spectrogram using Librosa and stores them in the "gtzan_mel" folder in your hw2 directory
* train_test.py: train models and test it 

Once you downloaded the files in your homework folder, run the feature extraction first:
```
$ python feature_extraction.py
```
If the run is successful (it takes some time), you will see that the "gtzan_mel" folder is generated and it contains the mel-spectrograms:

```
$ ls 
feature_extraction.py gtzan_mel readme.md train_test.py
```

Finally, run the traing and test code
```
$ python train_test.py
```

If the run is successful, it will display the losses along epochs, the average loss and test accuracy values.  

```
...
Learning rate : 1.6e-05
Epoch [24/50], Iter [10/88] loss : 1.4940
Epoch [24/50], Iter [20/88] loss : 0.3516
Epoch [24/50], Iter [30/88] loss : 0.4505
Epoch [24/50], Iter [40/88] loss : 0.0769
Epoch [24/50], Iter [50/88] loss : 1.4365
Epoch [24/50], Iter [60/88] loss : 0.3243
Epoch [24/50], Iter [70/88] loss : 0.3478
Epoch [24/50], Iter [80/88] loss : 0.5956
Average loss: 2.0249 
...

Epoch    23: reducing learning rate of group 0 to 3.2000e-06.
Learning rate : 3.2e-06
Early stopping

Average loss: 1.9201 

Test Accuracy: 0.4828 

```

## Improving Algorithms
Now it is your turn. You should improve the baseline code by developing your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* The first thing to do is to segment audio clips and generate more data. The baseline code utilizes the whole mel-spectrogram as an input to the network (e.g. 128x1287 dimensions). Try to make the network input between 3-5 seconds segment and average the predictions of the segmentations for an audio clip.
* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Patience value
    * Decreasing factor of learning rate 
    * Minibatch size
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* Also, you may try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* Furthermore, you can augment data using audio effects (if time permits)


## Deliverables
You should submit your Python code (.py files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion

