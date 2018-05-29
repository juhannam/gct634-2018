'''
result_visualizer.py

A file visualize result containing chromagram, annotation and prediction
Please check the 'IMPORT_DIR' before you run this code.
'''
import os
import numpy as np

import matplotlib.pyplot as plt
from librosa.display import specshow

IMPORT_DIR = './export/baseline_beatsync_result/'

#Function to decode onehot-labeled array into note
#Ex> 1  (C major) -----> [1 0 0 0 1 0 0 1 0 0 0 0] (C E G)
#Ex> 15 (D minor) -----> [0 0 1 0 0 1 0 0 0 1 0 0] (D F A)
def triad_decoder(prediction):
    num_keys = 12
    output = np.zeros((prediction.shape[0], 12))

    for i in range(prediction.shape[0]):
        key = int(prediction[i]%num_keys)
        chord = int(prediction[i]/num_keys)
        if chord == 0: #major chord
            output[i, key] = 1 #root
            output[i, (key+4)%num_keys] = 1 #major third
            output[i, (key+7)%num_keys] = 1 #perfect fifth
        elif chord == 1: #minor chord
            output[i, key] = 1 #root
            output[i, (key+3)%num_keys] = 1 #minor third
            output[i, (key+7)%num_keys] = 1 #perfect fifth

    return output

#Function to visualize chromagram, annotation and prediction from 'start_index' to 'start_index + index_length'
def chroma_result(chromagram, annotation, prediction, start_index, index_length):
    plt.figure(figsize=(19,11))
    plt.subplots_adjust(hspace = .02)
    sub1 = plt.subplot(3,1,1)
    specshow(chromagram[start_index:start_index+index_length].T, y_axis='chroma')
    sub1.set_ylabel('input chroma')
    sub2 = plt.subplot(3,1,2)
    specshow(annotation[start_index:start_index+index_length].T, y_axis='chroma')
    sub2.set_ylabel('annotation')
    sub3 = plt.subplot(3,1,3)
    specshow(prediction[start_index:start_index+index_length].T, y_axis='chroma')
    sub3.set_ylabel('prediction')
    plt.show()

def main():
    #Import
    chroma = np.load(os.path.join(IMPORT_DIR, 'chroma.npy'))
    prediction = np.load(os.path.join(IMPORT_DIR, 'prediction.npy'))
    annotation = np.load(os.path.join(IMPORT_DIR, 'annotation.npy'))

    prediction_decoded = triad_decoder(prediction)
    annotation_decoded = triad_decoder(annotation)

    #Visualize
    chroma_result(chroma, annotation_decoded, prediction_decoded, 0, 1000)

if __name__ == '__main__':
    main()
