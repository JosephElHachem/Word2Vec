# ABSTRACT
This code contains an implementation of the skip-gram model with negative sampling.

**To train:**
python skipGram.py --model path_to_save_model --text path_to_data

**To test**:
python skipGram.py --test --model mymodel.model --text path_to_data
-> prints out the similarities between words.
To save similarities in text file:
python skipGram.py --test --model mymodel.model --text path_to_data > results.txt

# PARAMETERS
- sentences       	-> preprocessed corpus. Type: list of lists, each sub list containing single word strings
- stopWords=[]    	-> list of strings
- n_epochs=20    		-> number of epochs for training
- start_from=None		-> path to start from, should contain: hidden_layer.npy, output_layer.npy, w2id.pkl, id2w.pkl, vocab.pkl
- save_path=None		-> path to save model at the end of training and each checkpoint_save. Saves: hidden_layer.npy, output_layer.npy, w2id.pkl, id2w.pkl, vocab.pkl
- checkpoint_save=1	-> save model parameters each "checkpoint_save" epochs, inside the "save_path"
- nEmbed=100		    -> length of the embedding layer
- negativeRate=5		-> size of negative ids sample
- winSize = 5		    -> window size for skip-gram
- minCount = 5		-> threshold of appreance of a words not to be deleted
- learning_rate=1e-2	-> learning rate for learning
- checkpoint=1000		-> print average loss each "checkpoint" sentences

# ARCHITECTURE

### GENERAL ARCHITECTURE
The word2vec model uses two matrices in order to find the embeddings.
The first matrix, `hidden_layer`, of shape (`vocab_size`, `nEmbed`),
represents the embedding of the words, while the second matrix, `output_layer` of shape (`nEmbe`, `vocab_size`),
is a 'context' matrix that will only be used for the training, and will be dropped at the end.
Both matrices are initialized with a uniform distribution over [-1,1].

Since the training process is slow, we introduce some flexibility of using traninig on previous sets.
When defining the model, we can choose to start from a previous training by giving the path to `start_from`.
Once the model is loaded, we expand our main objects to take into account the new observed words.
The previous mapping between words and tokens is conserved and expanded to the newly observed words.
The embedding layer and output layer are stacked with new matrixes initialized with a uniform distribution on [-1,1]

### PREPROCESSING
 1. Split sentences base on lines in .txt and on the following punctuations (., !, ?)
 2. Removing other punctuations
 3. Lower case words
 4. Replacing numbers with a <NUMBER> token
 5. Replacing negations with <NEGATIVE> token
 6. Removing STOP_WORDS using the spacy library
 7. Removing infrequent words
 8. Delete single word sentences
 
### TOKENIZATION
we iterate over the vocabulary and assign an int for each word, then define two mappings: w2id and id2w (word2id and id2word)

### NEGATIVE SAMPLING
The negativeRate is fixed to 5.
At each step, we sample 5 distinct words, different from the current word and current context word.
The sampling distribution is the one from the original paper explained here  http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/,
where the probability of each word is its occurrence raised to the power of 0.75 and then normalize.
In order to sample using this distribution, we define a unigram table of length `1e8`
that respects the probabilities defined above, and we sample uniformly from it.

### LOSS FUNCTION
Loss function used is the logsoftmax

### SIMILARITY
We use a cosine distance shifted by +1 and normalized to 1 as our similarity measure.

### MISSING WORDS
For calculating the similarity when one word in unknown, we a number from a guassian distribution centered N(0.5, 0.1)

### EVALUATION
For evaluation, we observe the convergence of the loss.
We also define a function called most_similar that takes as input a word and a number k and returns the k most similar words in the voca.
We also define a function called test_model that takes as input a path for a ground truth and computes the correlation between results of similarities.

### HYPERPARAMETERS CHOICE
learning rate: after a few quick experiences, learning_rate = 1e-2
For the rest of parameters, as we did not have time to experiment since the model takes a considerable time to train, we used standard choices:
embedding dimension: 100
window_size=5
negativeRate=5 
minCount=5


# REFERENCES
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributedrepresentations of words and phrases and their compositionality, 2013
* Eric Kim. Optimize Computational Efficiency of Skip-Gram with Negative Sampling. 26 May 2019. https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling
* Chris McCormick. Word2vec Tutorial Part 2: Negative Sampling. 11 Jan 2017. http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
* Yoav Goldberg and Omer Levy. word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method, 2014
