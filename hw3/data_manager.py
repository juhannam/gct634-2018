'''
data_manager.py

A file that related with managing data.
'''
import os
import numpy as np
import glob as gl

#Class for saving seperated train, test, and valid dataset
class Data(object):
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid

#Class for saving chroma, chord, and beat information
class Info(object):
    def __init__(self, chroma, chord, beat):
        self.chroma = chroma
        self.chord = chord
        self.beat = beat

#Loading entire data from given directory and return list of numpy
def load_data(data_dir):
    data_list = []
    filelist = gl.glob(os.path.join(data_dir, '*.npy'))
    for file in sorted(filelist):
        data = np.load(file)
        data_list.append(data)

    return data_list

#Converting list into batched numpy and the obtained shape is (number of batch, batch size, extra dimension)
def batch_dataset(dataset, batch_size):
    batch_data_list = []
    for data in dataset:
        num_batch = int(np.floor(data.shape[0]/batch_size))
        batch_data = data[:num_batch*batch_size]
        if data.ndim == 1:
            batch_data = batch_data.reshape(num_batch,batch_size)
        elif data.ndim == 2:
            batch_data = batch_data.reshape(num_batch,batch_size,data.shape[1])
        batch_data_list.append(batch_data)

    return np.concatenate(batch_data_list, axis=0)

#Converting chroma and chord into beat-synchronous chroma and chord using beat information
def beatsync(chroma, chord, beat):
    beat_chroma_list = []
    beat_chord_list = []

    for i in range(len(chroma)):
        beat_chroma = np.zeros((beat[i].shape[0] - 1, chroma[i].shape[1]))
        for j in range(beat[i].shape[0] - 1):
            beat_chroma[j,:] = np.mean(chroma[i][beat[i][j]:beat[i][j+1],:], axis=0)

        beat_chord = np.zeros(beat[i].shape[0] - 1).astype(int)
        for j in range(beat[i].shape[0] - 1):
            beat_chord[j] = np.bincount(np.squeeze(chord[i][beat[i][j]:beat[i][j+1]])).argmax()

        beat_chroma_list.append(beat_chroma)
        beat_chord_list.append(beat_chord)

    return beat_chroma_list, beat_chord_list

#Function to preprocess raw data into seperated and batched dataset and chroma, chord and beat information
#'frame' mode for frame level chroma and 'beatsync' mode for beat-synchronous chroma
def preprocess(dataset_dir, batch_size, train_ratio=0.6, test_ratio=0.2, mode='frame'):
    chroma = load_data(os.path.join(dataset_dir, 'chroma'))
    chord = load_data(os.path.join(dataset_dir, 'chord'))
    if mode == 'frame':
        x, y = chroma, chord

    elif mode == 'beatsync':
        beat = load_data(os.path.join(dataset_dir, 'beat'))
        x, y = beatsync(chroma, chord, beat)

    train_size = int(np.round(len(x)*train_ratio))
    test_size = int(np.round(len(x)*test_ratio))

    x_train = batch_dataset(x[:train_size], batch_size)
    x_test = batch_dataset(x[train_size:train_size+test_size], batch_size)
    x_valid = batch_dataset(x[train_size+test_size:], batch_size)

    y_train = batch_dataset(y[:train_size], batch_size)
    y_test = batch_dataset(y[train_size:train_size+test_size], batch_size)
    y_valid = batch_dataset(y[train_size+test_size:], batch_size)

    chroma_test = chroma[train_size:train_size+test_size]
    chord_test = chord[train_size:train_size+test_size]
    if mode == 'frame':
        beat_test = None

    elif mode == 'beatsync':
        beat_test = beat[train_size:train_size+test_size]

    return Data(x_train, x_test, x_valid), Data(y_train, y_test, y_valid), Info(chroma_test, chord_test, beat_test)

#Function to obtain frame level accuracy for both frame level and beat-synchronous chroma
#'frame' mode does not use info and batch size
#'beatsync' mode use info and batch size, especially use timing information in info.chord and info.beat
def frame_accuracy(annotation, prediction, info=None, batch_size=None, mode='frame'):
    if mode == 'beatsync':
        current_beat = 0
        prediction_frame_list = []
        for i in range(len(info.chroma)): #Iteration of songs
            num_song_frame = int(info.chroma[i].shape[0]/batch_size)*batch_size
            prediction_song_frame = np.zeros(num_song_frame)
            num_beat = int((info.beat[i].shape[0] - 1)/batch_size)*batch_size
            for j in range(num_beat):
                prediction_song_frame[info.beat[i][j]:info.beat[i][j+1]] = prediction[current_beat+j]
            prediction_frame_list.append(prediction_song_frame)
            current_beat += num_beat

        prediction = np.concatenate(prediction_frame_list)

    accuracy = 1-np.count_nonzero(prediction - annotation)/float(prediction.shape[0])

    return accuracy, prediction

