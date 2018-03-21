# Homework #1: Musical Instrument Classification
Musical instrument classification is a fundamental task in understanding music by computers. Your first mission is developing your own algorithm based on the traditional machine learning approach. Specifically, the goals of this homework are as follows: 
- Experiencing the whole pipeline of a machine learning task: data preparation, feature extraction, training learning models and evaluation 
- Using the Librosa and Scikit-learn libraries in practice 
- Analyzing different characteristics of musical instrument tones and extracting them in a numerical form

## Dataset
We use a subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) which is a large collection of musical instrument tones from the Google Magenta project. The subset has 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. For our expriment, it is split into training, validation and test sets. For each class, the training set has 100 audio samples and both validation and test sets have 20 audio samples. You can download the subset from our course page in [KLMS](http://klms.kaist.ac.kr/). 

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

```
$ cd dataset
$ ls 
test test_list.txt train train_list.txt valid valid_list.txt
$ cd ..      # go back to your home folder for next steps
```

## Baseline Code
The source code of the baseline algorithm is provided so that you can easily start with the homework and also compare your own algorithm to it in performance. The baseline model extracts MFCC, summarizes them by taking temporal average for each audio file and use a linear SVM model for classification.  

From the homework git repository, download the following Python files. 

* feature_extraction.py: loads audio files, extracts MFCC features using Librosa and stores them in the "mfcc" folder
* feature_summary.py: contains functions to summarize the extracted MFCC features
* train_test.py: train models and test it 

Once you downloaded the files in your homework folder, run the feature extraction first:
```
$ python feature_extraction.py
```
If the run is successful (it takes some time), you will see that the "mfcc" folder is generated and it contains the extracted features:

```
$ ls 
dataset mfcc feature_extraction.py feature_summary.py train_test.py
```

Finally, run the traing and test code
```
$ python train_test.py
```

If the run is successful, it will display the validation and test accuracy values.  

```
validation accuracy = 43.5 %
validation accuracy = 45.0 %
validation accuracy = 41.5 %
validation accuracy = 43.5 %
validation accuracy = 41.0 %
validation accuracy = 10.0 %
test accuracy = 43.0 %
```

## Improving Algorithms
Now it is your turn. You should improve the baseline code by developing your algorithm. There are many ways to improve it. The followings are possible ideas: 

* Try different classifiers: k-NN, SVM with nonlinear kernels, MLP, GMM, ...
* Adjust the size of context window of MFCCs: taking a small size of context window will increase the number of examples for classifiers 
* Codebook-based feature summarization
* Try different MFCC parameter settings: mel-bin size and DCT size
* Try to add delta and double-delta of MFCCs
* Try to add other audio features: think about what determines the unique timbre of musical instruments. For example, attack and sustain, harmonic and noise, temporal envelope and so on.  

## Deliverables
You should submit your Python code (.py files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion

