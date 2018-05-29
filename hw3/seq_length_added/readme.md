# Homework #3: Sequence length applied
Now data processed from data_manager.py has sequence length dimension. batch_dataset in data_manager.py pads  previous frame, of which length is sequence length, to current frame. Thus the length, or number of batch dimension remains same.

## Things changed
If you already improved algorithm a lot from the original baseline code please refer this section to apply the changes.

* main.py
  * SEQ_LENGTH
  * data_manager.preprocess - SEQ_LENGTH added
  * wrapper = Wrapper(~)
  * chroma_test = chroma_test.reshape(~)

* data_manager.py
  * batch_dataset
  * preprocess - input of the function, batch_dataset for x

* model_archive.py
  * RNN - input for fully connected layer, output.squeeze() removed

* model_wrapper.py
  * run_model - input for a model
