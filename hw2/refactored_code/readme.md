## Baseline Code
The refactored code of the baseline algorithm is provided so that one can start the homework with compact coding style. 


From the homework git repository, download the following Python files. 

* feature_extraction.py: loads audio files, extracts mel-spectrogram using Librosa and stores them in the "gtzan_mel" folder in your hw2 directory
* main.py: run model 
* model.py: model class
* train.py: fit and eval function
* data_loader.py: dataloader class
* preprocessing.py: load label function

Once you downloaded the files in your homework folder, run the feature extraction first:
```
$ python feature_extraction.py
```
If the run is successful (it takes some time), you will see that the "gtzan_mel" folder is generated and it contains the mel-spectrograms:

```
$ ls 
feature_extraction.py gtzan_mel readme.md data_loader.py main.py model.py preprocessing.py train.py
```

Finally, run the traing and test code
```
$ python main.py --gpu_use=0 --which_gpu=0
```
If you use gpu, put 1 in gpu_use. 

If the run is successful, it will display the losses along epochs, the average loss and test accuracy values.  

```
Learning rate : 1.6e-05
Epoch [27/50], Iter [10/88] loss : 0.8006
Epoch [27/50], Iter [20/88] loss : 0.1059
Epoch [27/50], Iter [30/88] loss : 0.0665
Epoch [27/50], Iter [40/88] loss : 0.2308
Epoch [27/50], Iter [50/88] loss : 0.1968
Epoch [27/50], Iter [60/88] loss : 0.0850
Epoch [27/50], Iter [70/88] loss : 0.3561
Epoch [27/50], Iter [80/88] loss : 0.0296
Average loss: 2.1946 

Epoch    26: reducing learning rate of group 0 to 3.2000e-06.
Learning rate : 3.2e-06
Early stopping


--- 196.308371067 seconds spent ---
Average loss: 2.1693 

Test Accuracy: 0.4931 

```

