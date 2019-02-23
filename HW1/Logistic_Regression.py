#!/usr/bin/env python
# coding: utf-8

# In[36]:


from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from collections import Counter


from scipy.sparse import csr_matrix


import string
import matplotlib.pyplot as plt





# In[12]:





# In[13]:




# In[449]:






# In[10]:


stop_word = ['i', 'me', 'my', 'myself', 'we', 'our','u','ur','re', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


def tokenize(dataframe):
    lst_of_token = []
    for i in dataframe:     
        lst_of_token += tokenstring(i)
    return lst_of_token

def tokenstring(i):
    lst_of_token = []
    i= i.lower()
    for char in string.punctuation:
        if char == "'":
            i = i.replace(char, "")
        else:
            i = i.replace(char, " ")
    text = list(i.split())
    temp = []
    for word in text:
        if word not in stop_word:
            temp.append(word)
    lst_of_token += temp
    return lst_of_token


# In[14]:
    

df1 = pd.read_csv("train.csv")
df2 = df1[df1.Insult == 0]
df3 = df1[df1.Insult == 1]
df0 = df1

lst_of_0_token = tokenize(df2.Comment)
counter_0 = Counter(lst_of_0_token)

lst_of_1_token = tokenize(df3.Comment)
counter_1 = Counter(lst_of_1_token)

lst_of_all_token = tokenize(df1.Comment)
counter_all = Counter(lst_of_all_token)


# In[15]:


all_dict = counter_all
all_dict = dict.fromkeys(all_dict,0)
# print(all_dict)
all_dict['BIAS'] = 1


# In[16]:



def make_comment_dict_lst(df_name):
    lst_of_comment_dict = []
    for i in df_name.Comment:
        lst_of_comment = tokenstring(i)
        count_comment = Counter(lst_of_comment)
        count_comment['BIAS'] = 1
        lst_of_comment_dict.append(count_comment)
#     print(lst_of_comment_dict[:1])
    return lst_of_comment_dict


# In[17]:


comment_dict_lst = make_comment_dict_lst(df1)


# In[18]:




# In[19]:



def make_row_matrix(lst):

    row_array_dict = {}
    n = 0
    for i in lst:
        all_dict = counter_all
        all_dict = dict.fromkeys(all_dict,0)
        all_dict['BIAS'] = 1
        for word in dict(i):
    #         print(word)
            all_dict[word] = i[word]
        new_array = np.array(list(all_dict.values()))
    #     print(new_array)
        sparse_matrix = csr_matrix(new_array)
    #     print(sparse_matrix)
        row_array_dict[n] = sparse_matrix
        n += 1
#         print(array_dict)
    return row_array_dict


# In[20]:


row_array_dict = make_row_matrix(comment_dict_lst)


# In[21]:


# print(row_array_dict)


# In[458]:


# In[22]:


def make_X_dict(array_dict, b):
    X_dict = {}
    n = 0
    for array in array_dict.values():
        sumxb = array.dot(b)
        X_dict[n]  =sumxb
        n+=1
    return X_dict

    


# In[23]:



# In[28]:


def make_true_value(df_name):
    true_lst = []
    for row in df_name.Insult:
        true_lst.append(row)
    return true_lst


# In[29]:


true = make_true_value(df1)

# In[30]:
def sigmoid(X):
    result = 1 / (1 + np.exp(-X))
    return result

# In[31]:
def log_likelihood(xi, yi, b):
    result  = np.sum(yi*(xi.dot(b)) - np.log(1+ np.exp(xi.dot(b))))
    return result
# In[31]:


def predict(xi,beta):
    prediction = float(sigmoid(xi.dot(beta)))
    if prediction > 0.5:
        pro = 1
    else:
        pro =0
    return pro
    
# In[42]:
def compute_gradient(xi, yi, b):
    error = yi - sigmoid(xi.dot(b))
    gradient = xi.T.dot(error)
    return gradient

# In[43]:
#
     

# In[43]:


log_lst = []
round_lst = [ ]
def logistic_regression(alpha, x, y,steps ):
    
    b = np.zeros((len(all_dict),1))
    round_num = 0
    while round_num < steps:
        print("round", round_num)
        for loop in range(len(df1.Comment)):
            i = np.random.random_integers(3946)
            ara = alpha*compute_gradient(x[i],y[i],b)
            b = b + ara
            if round_num%3 == 0:
                log_lst.append(log_likelihood(x[i], y[i], b))
                round_lst.append(round_num)
        round_num += 1
    return b





b_trained = logistic_regression(5*10**(-5), row_array_dict,true, 300)
# In[ ]:
plt.plot(round_lst,log_lst)
plt.show()
# In[ ]:
# log_lst_1 = log_lst
# round_lst_1 = round_lst
# # In[ ]:
# test_data = pd.read_csv("test.csv")


# comment_lst_test = []
# for row in test_data.Comment:
#     row_lst_word = []
#     lst_of_word_test = tokenstring(row)
#     for word in lst_of_word_test:
#         if word in list(all_dict.keys()):
#             row_lst_word.append(word)
#     count_comment_word_test = Counter(row_lst_word)
#     count_comment_word_test['BIAS'] = 1
#     comment_lst_test.append(count_comment_word_test)

# # In[ ]:
# row_array_dict_test = make_row_matrix(comment_lst_test)


# # In[ ]:
# pro_lst = []
# for xi in row_array_dict_test.values():
#     pro_lst.append(predict(xi, b_trained))

# test_data.Insult = pro_lst



# # In[ ]:


# test_data.to_csv("test_fin.csv", index = False)