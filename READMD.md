# Briefing
This code contains an implementation of the skip-gram model with negative sampling.
To train:
python skipGram.py --model path_to_save_model --text path_to_data

To test:
python skipGram.py --test --model mymodel.model --text path_to_data
-> prints out the similarities between words.
To save similarities in text file:
python skipGram.py --test --model mymodel.model --text path_to_data > results.txt

# Preprocessing
 1. Removing punctuations
 2. Replacing numbers with a <NUMBER> token
 3. Replacing negations with <NEGATIVE> token
 4. Removing infrequent words
 5. Removing STOP_WORDS using the spacy library

####################################################################
PARAMETERS
##################################
sentences       	-> preprocessed corpus. Type: list of lists, each sub list containing single word strings
stopWords=[]    	-> list of strings
n_epochs=20    		-> number of epochs for training
start_from=None		-> path to start from, should contain: hidden_layer.npy, output_layer.npy, w2id.pkl, id2w.pkl, vocab.pkl
save_path=None		-> path to save model at the end of training and each checkpoint_save. Saves: hidden_layer.npy, output_layer.npy, w2id.pkl, id2w.pkl, vocab.pkl
checkpoint_save=1	-> save model parameters each "checkpoint_save" epochs, inside the "save_path"
nEmbed=100		-> size of the embedding layer
negativeRate=5		-> size of negative ids sample
winSize = 5		-> window size for skip-gram
minCount = 5		-> threshold of appreance of a words not to be deleted
learning_rate=1e-2	-> learning rate for learning
checkpoint=1000		-> print average loss each "checkpoint" sentences

####################################################################
GENERAL ARCHITECTURE
##################################
Since the training process is slow, we introduce some flexibility of using traninig on previous sets.
When defining the model, we can choose to start from a previous training by giving the path to "start_from".
Once the model is loaded, we expand our main objects to take into account the new observed words.
The previous mapping between words and tokens is conserved and expanded to the newly observed words.
The embedding layer and output layer are stacked with new matrixes initialized with a uniform distribution on [-1,1]

####################################################################
PREPROCESSING:
##################################
Before passing to the model: (using text2sentences)
1. Remove punctuation (replaced with a single space)
2. Lower case words
3. Split sentence by space and keep only fully alphabetic words

After passing to the model: (using self.remove_unfrequent)
4. Removing unfrequent words (appreance less then a threshold fixed to 5)
-> The model doesn't have enough data to learn the embedding
5. Removing stop words (ex: the, would, should, then, what, etc. ). The list of stop words is in the SkipGram.py
-> Such words do not contain context information which impacts the training for other words. Also they are too frequent which will slow down the learning for other words.
6. Deleting single word sentences after deletion of words in the steps above.

####################################################################
TOKENIZATION
##################################
we iterate over the vocabulary and assign an int for each word, then define two mappings: w2id and id2w (word2id and id2word). However, the training set is not tokenized.

####################################################################
INITIALIZATON
##################################
We initialize the embedding layer and the context layer with a uniform distribution on [-1,1].

####################################################################
NEGATIVE SAMPLING
##################################
The negativeRate is fixed to 5.
At each step, we sample 5 distinct words, different from the current word and current context word.
The sampling distribution is the one from the slides (cours2, slide30), where we count the occurence of each word and raise it to the power of 0.75, then normalize the counts.
The numpy function to sample from a self defined distribution (numpy.random.choice) is extremely slow, and can harshly impact the speed of the algorithm if called at each iteration.
To overcome this issue, we use the function once in the beginning to generate a new set using the tokenized vocabulary and the frequencies, the set length being the same one as the corpus length.
This will take some memory space but certainly not as much as out corpus since out new set is one dimensional and each element is an integer.

####################################################################
FORWARD/BACKWARD
##################################
We only calculate the forwad and backward passes for the contextId and negativeIds.
We use some matrix multiplications to speed up calculations.
Check out the code.

####################################################################
LOSS FUNCTION
##################################
The loss is equal to th sum of the differences between the output and the expected probabilities of the context word and words from negative sampling.

####################################################################
SIMILARITY
##################################
We use a cosine distance shifted by +1 and normalized to 1 as our similarity measure.

####################################################################
MISSING WORDS
##################################
For calculating the similarity when one word in unknown, we a number from a guassian distribution centered N(0.5, 0.1)

####################################################################
EVALUATION
##################################
For evaluation, we observe the convergence of the loss.
We also define a function called most_similar that takes as input a word and a number k and returns the k most similar words in the voca.
We also define a function called test_model that takes as input a path for a ground truth and computes the correlation between results of similarities.

####################################################################
HYPERPARAMETERS CHOICE
##################################
learning rate: after a few quick experiences, learning_rate = 1e-2
For the rest of parameters, as we did not have time to experiment since the model takes a considerable time to train, we used standard choices:
embedding dimension: 100
window_size=5
negativeRate=5 
minCount=5

####################################################################
DIFFICULTIES
##################################
Our main difficulty in this project is that it took us some time to figure out that the code was too slow because we were using the np.random.choice function at each iteration.
Numpy has the reputation of being fast but this is only true when vectorized. This issue did not leave us enough time to do a proper training on large corpuses and do the experiments we wanted to do.


####################################################################
REFERENCES
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributedrepresentations of words and phrases and their compositionality, 2013

* Eric Kim. Optimize Computational Efficiency of Skip-Gram with Negative Sampling. 26 May 2019. https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling

* Chris McCormick. Word2vec Tutorial Part 2: Negative Sampling. 11 Jan 2017. http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

* Yoav Goldberg and Omer Levy. word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method, 2014


### End
Author:
Joseph El Hachem
