#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
nltk.download('punkt')
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords


# In[3]:


stopwords = set(stopwords.words('english'))

# stopwords


# In[4]:


stopwords_new = [word for word in stopwords if word != 'do']


# In[5]:


import csv
test_word = []
with open('intrinsic-test.tsv') as f:
    test_data  = csv.reader(f, delimiter='\t')
    for row in test_data:
        test_word.append(row[1])
        test_word.append(row[2])


# In[6]:


test_word_new = test_word[2:]


# In[7]:


test_word_new


# In[8]:


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from


# In[9]:


def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = True
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    # ... load in first 15,000 rows of unlabeled data file.  You can load in
    # more if you want later (and should do this for the final homework)
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
#     fullconts = fullconts[1:15000]  # (TASK) Use all the data for the final submission
    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]
#     print(fullconts)





    print ("Generating token stream...")
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.
    fullrec = nltk.word_tokenize(fullconts[0])
#     print(fullrec)
    min_count = 50
    origcounts = Counter(fullrec)
#     print(origcounts)





    print ("Performing minimum thresholding..")
#     #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
#     #... replace other terms with <UNK> token.
#     #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)



#     #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec_filtered = ['<UNK>' if (origcounts[word] < min_count) & (word not in test_word_new)  else word for word in fullrec]
    fullrec_filtered2 = []
    for word in fullrec_filtered:
        if word not in stopwords_new:
            fullrec_filtered2.append(word)
#     print(fullrec_filtered)
    fullrec = fullrec_filtered2
    wordcounts = Counter(fullrec)





    print ("Producing one-hot indicies")
#     #... (TASK) sort the unique tokens into array uniqueWords
#     #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
#     #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = sorted([word for word in wordcounts])#... fill in
    
#     print(uniqueWords)
    wordcodes = {}
    for word in uniqueWords:
#         one_hot = np.zeros(len(uniqueWords),)
#         wordIndex = uniqueWords.index(word)
#         one_hot[wordIndex] = 1
#         wordcodes[word] = one_hot
        wordcodes[word] = uniqueWords.index(word)
#     print(wordcodes)
#     wordcodes = #... fill in

    




#     #... close input file handle
    handle.close()

    fullrec  = [wordcodes[word] for word in fullrec]

#     #... store these objects for later.
#     #... for debugging, don't keep re-tokenizing same data in same way.
#     #... just reload the already-processed input data with pickles.
#     #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


#     #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
#     print(fullrec)
    return fullrec
    




# In[10]:


@jit
def sigmoid(x):
    return 1.0/(1+np.exp(-x))


# In[11]:


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
#     global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0
    table_size = 1e7
#     print(train_data)
    print ("Generating exponentiated count vectors")
    #... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = []
    for word in uniqueWords:
        if word != "<UNK>":
            exp_count_array.append(wordcounts[word]**exp_power) #... fill in
#     print(exp_count_array)
    max_exp_count = sum(exp_count_array)
    

    print ("Generating distribution")

    #... (TASK) compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = [count_freq / max_exp_count for count_freq in exp_count_array] #... fill in
#     print(prob_dist)




    print ("Filling up sampling table")
    #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.
    table_size = 1e7
    cumulative_dict = {}#... fill in
    
#     freqTempList = [freq*table_size for freq in prob_dist] 
# #     keyTempList = [i for i in range(table_size)]
#     valueTempList = []
#     for i,freq in freqTempList:
#         for j in freq:
#             valueTempList.append(train_data[i])
    begin = 0
    for i,prob in enumerate(prob_dist):
        freq = int(round(prob*table_size))
        for j in range(freq):
            cumulative_dict[j+begin] = i
        begin += freq
    
        
#     cumulative_dict = dict(zip(keyTempList,valueTempList))
            
            
        
    




    return cumulative_dict


# In[12]:


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    while len(results) < num_samples:
        sample = samplingTable[random.randint(0, len(samplingTable)-1)]
#         while samplingTable[random.randint(0, len(samplingTable)-1)] = context_idx:
#             sample = samplingTable[random.randint(0, len(samplingTable)-1)]
#         results.append(sample)
        if sample != context_idx:
            results.append(sample)


    return results


# In[13]:


@jit(nopython=True)
def performDescent(learning_rate, center_token, context_index,W1,W2,negative_indices):# delete the sequence_chars and num_samples
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
#     for k in range(0, len(sequence_chars)):
        #... (TASK) implement gradient descent. Find the current context token from sequence_chars
        #... and the associated negative samples from negative_indices. Run gradient descent on both
        #... weight matrices W1 and W2.
        #... compute the total negative log-likelihood and store this in nll_new.
    
    H = W1[center_token].T
    y_predict = sigmoid((W2[:,context_index].T.dot(H)))
    error = y_predict - 1
    
    total = error*W2[:,context_index]
    W2[:,context_index] = W2[:,context_index] - learning_rate*error*H
    nll_new += -np.log(sigmoid((W2[:,context_index].T.dot(H))))
    
    for index in negative_indices:
        y_predict_negative = sigmoid((W2[:,index].T.dot(H)))
        error_negative = y_predict_negative - 0
        
        total += error_negative*W2[:,index]
        W2[:,index] = W2[:,index] - learning_rate*error_negative*H 
        
        nll_new += -np.log(sigmoid((-W2[:,index].T.dot(H))))
        
    W1[center_token] = W1[center_token] - learning_rate*total.T
    

    

    



    return nll_new


# In[24]:


def trainer(curW1 = None, curW2= None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    try:
        if curW1 == None:
            np_randcounter += 1
            W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
            W2 = np.random.uniform(-.5, .5, size=(hidden_size, vocab_size))
            epochs = 5
            num_samples = 2
            learning_rate = 0.05
            nll = 0
            iternum = 0



            indexUNK = wordcodes['<UNK>']
            for j in range(0,epochs):
                print ("Epoch: ", j)
                prevmark = 0

                #... For each epoch, redo the whole sequence...
                for i in range(start_point,end_point):
                    iternum += 1

                    if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                        print ("Progress: ", round(prevmark+0.1,1))
                        prevmark += 0.1
                    if iternum%10000==0:
                        print ("Negative likelihood: ", nll)
                        nll_results.append(nll)
                        nll = 0


                    #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
                    if mapped_sequence[i] != indexUNK :
                        center_token = mapped_sequence[i]
        #                 print(mapped_sequence[:10])
        #                 print(center_token)#... fill in
                    #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
                    else:
                        continue


        #             iternum += 1
                    #... now propagate to each of the context outputs
                    for k in range(0, len(context_window)):

                        #... (TASK) Use context_window to find one-hot index of the current context token.
                        context_index = mapped_sequence[context_window[k] + i]#... fill in



                        #... construct some negative samples
                        negative_indices = generateSamples(context_index, num_samples)

                        #... (TASK) You have your context token and your negative samples.
                        #... Perform gradient descent on both weight matrices.
                        #... Also keep track of the negative log-likelihood in variable nll.
                        nll = performDescent(learning_rate, center_token, context_index,W1,W2,negative_indices)

            for nll_res in nll_results:
                print (nll_res)
    except:
#         ... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
               
                
                




    return [W1,W2]



# In[15]:


def load_model():
    handle = open("saved_W1.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]



# In[ ]:





# In[16]:






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle = True)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle = True)
    handle.close()


# In[17]:


word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)


# In[ ]:





# In[ ]:





# In[51]:


#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [word_seq[0], # suffix averaged
    embeddings[wordcodes[word_seq[1]]]]
    vector_math = vectors[0]+vectors[1]
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.
    tempDict = {}
    resultDict = {}
    for word in uniqueWords:
        vector_pre = embeddings[wordcodes[word]]
        tempDict[word] = 1 - spatial.distance.cosine(vector_pre, vector_math)
    outputs = sorted(tempDict, key = tempDict.get, reverse = True)
    for word in outputs[:10]:
        resultDict[word] = tempDict[word]
    return resultDict


# In[48]:


#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] + vectors[2] # + vectors[3] = 0
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.
    tempDict = {}
    resultDict = {}
    for word in uniqueWords:
        vector_pre = embeddings[wordcodes[word]]
        tempDict[word] = 1 - spatial.distance.cosine(vector_pre, vector_math)
    outputs = sorted(tempDict, key = tempDict.get, reverse = True)
    for word in outputs[:10]:
        resultDict[word] = tempDict[word]
    return resultDict




# In[20]:



#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................

from scipy import spatial
def get_neighbors(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    tempDict = {}
    resultDict = {}
    #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    targetIndex = wordcodes[target_word]
    for i in range(len(uniqueWords)):
        if i != targetIndex:
            tempDict[uniqueWords[i]] = 1 - spatial.distance.cosine(word_embeddings[i], word_embeddings[targetIndex])
    
        
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
    outputs = sorted(tempDict, key = tempDict.get, reverse = True)
    for word in outputs[:10]:
        resultDict[word] = tempDict[word]
    
    
    
    return resultDict
    


# In[21]:


def count_similiarity(word1, word2):
    global word_embeddings, wordcodes
    word1_index = wordcodes[word1]
    word2_index = wordcodes[word2]
    result = 1 - spatial.distance.cosine(word_embeddings[word1_index], word_embeddings[word2_index])
    return result


# In[25]:


filename = 'unlabeled-data.txt'
#... load in the file, tokenize it and assign each token an index.
#... the full sequence of characters is encoded in terms of their one-hot positions

fullsequence= loadData(filename)
print ("Full sequence loaded...")
#print(uniqueWords)
#print (len(uniqueWords))



#... now generate the negative sampling table
print ("Total unique words: ", len(uniqueWords))
print("Preparing negative sampling table")
samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


#... we've got the word indices and the sampling table. Begin the training.
#... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
#... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
#... ... and uncomment the load_model() line

train_vectors(preload= True)
[word_embeddings, proj_embeddings] = load_model()


# In[36]:


import csv

with open('intrinsic-test.tsv','r') as tsvfile:
    with open('result.csv','w') as re:
        writer = csv.writer(re, delimiter=',')
        writer.writerow(['id', 'sim'])
        test_data  = csv.reader(tsvfile, delimiter='\t', )
        next(test_data, None)
        for row in test_data:
            try:
                result = count_similiarity(row[1], row[2])
                index = row[0]
                writer.writerow([index, result])
            except:
                writer.writerow([row[0],0])


        
    


# In[37]:


import pandas as pd


# In[38]:


result = pd.read_csv('result.csv')


# In[39]:


result.to_csv('result2.csv', index = False)


# In[ ]:





# In[41]:


def prediction(word):
    
    predict = []
    preds = get_neighbors(word)
    for pred in preds:
        tempdict = {}
        simi = count_similiarity(word, pred)
        tempdict['word'] = pred
        tempdict['score'] = simi
        predict.append(tempdict)
    return predict


# In[52]:

print(sys.argv[1])
if __name__ == '__main__':
    if len(sys.argv)==2:
        filename = sys.argv[1]
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))



        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        train_vectors(preload=True)
        [word_embeddings, proj_embeddings] = load_model()








        #... we've got the trained weight matrices. Now we can do some predictions
        targets = ["good", "bad", "scary", "funny"]
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (prediction(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
            print ("\n")



        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        print (analogy(["son", "daughter", "man"]))
        print (analogy(["thousand", "thousands", "hundred"]))
        # print (analogy(["amusing", "fun", "scary"]))
        print (analogy(["terrible", "bad", "amazing"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.

        s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
        others = [["types", "type"],
                  ["ships", "ship"],
                  ["values", "value"],
                  ["walls", "wall"]]
                  # ["spoilers", "spoiler"]]
        for rec in others:
            s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
        s_suffix = np.mean(s_suffix, axis=0)
        print (morphology([s_suffix, "techniques"]))
        print (morphology([s_suffix, "sons"]))
        print (morphology([s_suffix, "secrets"]))






    else:
        print ("Please provide a valid input filename")
        sys.exit()


# In[ ]:




