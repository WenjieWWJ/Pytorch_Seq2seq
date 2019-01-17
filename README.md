# Pytorch_Seq2seq
The sequence-to sequence model implemented by pytorch

modified from pytorch tutorial(https://pytorch.org/tutorials/)

## Requirements
* python3
* pytorch >= 1.0.0

## Code

### Data Input

Change the dir and filenames in main.py
```
######################################################################
# Load Data

Min_word_count = 20 # the min freq of words

voc, train_pairs = loadTrainingData("./data", "train_post.txt", "train_response.txt", "./data")
voc.trim(Min_word_count) # remove the word from the vocab when the word freq < min_word_count

print("Vocab Size: {}".format(voc.num_words) )
valid_pairs = loadData("./data", "valid_post.txt", "valid_response.txt")

```
* train_post.txt: "\n" splits the samples.
* Every line in "train_response.txt" is the reply to the corresponding line in "train_post.txt".
* loadTrainingData(data_dir, post_name, response_name, save_dir) is used to load training data and construct vocab. save_dir denotes the dir of saved vocab.txt.
* loadData(data_dir, post_name, response_name) helps to load pair-wise data.

### Parameters

Modify the parameters in main.py

```
######################################################################
# Configures of the model

model_name = 'seq2seq_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

corpus_name = "mmd_prod_des"
save_dir = './save'

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 40000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 400000
print_every = 10
save_every = 500
valid_every = 100
```

### Training
Then you can load data and start training! Also can change the mode to save parameters in trainIters().
```
python3 main.py
```
```
# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, train_pairs, valid_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
```


### Inference

When the training is done, inference the results with the input from the terminal.

```
######################################################################
# Starting Inference
print("Starting Inference!")

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
```

## To Do List
* Beam Search Decoder
* Batched Inference 


