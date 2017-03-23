from numpy import *
from pylab import *
from math import *
import re
import time
import codecs

file = codecs.open('dataset.txt','r','utf-8')
documents = [document.strip() for document in file] 
file.close()

N = len(documents)

K = 10

#==============================================================================
#
#==============================================================================


wordCount = {}

wordCountPerDocument = [];

punctuationRegex = '[,.;"?!#-_…()`|“”‘]+'

stopwords = ['a','an', 'after', 'also', 'they', 'man', 'zou', 'can', 'and', 'as', 'up', 'soon', 'be', 'being', 'but', 'by', 'd', 'for', 'from', 'he', 'her', 'his', 'in', 'is', 'more', 'of', 'often', 'the', 'to', 'who', 'with', 'people', 'or', 'it', 'that', 'its', 'are', 'has', 'was', 'on', 'at', 'have', 'into', 'no', 'which']

for d in documents:
    words = d.split()
    wordCountCurrentDoc = {}
    for w in words:
      
        w = re.sub(punctuationRegex, '', w.lower())
        if len(w)<=1 or re.search('http', w) or re.search('[0-9]', w) or w in stopwords:
            continue
       
        if w in wordCount:
            wordCount[w] += 1
        else:
            wordCount[w] = 1
        if w in wordCountCurrentDoc:
            wordCountCurrentDoc[w] += 1
        else:
            wordCountCurrentDoc[w] = 1
    wordCountPerDocument.append(wordCountCurrentDoc);


#==============================================================================
# 
#==============================================================================

dictionary = {}

dictionaryReverse = {}

index = 0;
for word in wordCount.keys():
    if wordCount[word] > 1:
        dictionary[word] = index;
        dictionaryReverse[index] = word;
        index += 1;

M = len(dictionary)  

#==============================================================================
# 
#==============================================================================

X = zeros([N, M], int8)

for word in dictionary.keys():
    j = dictionary[word]
    for i in range(0, N):
        if word in wordCountPerDocument[i]:
            X[i, j] = wordCountPerDocument[i][word];


#==============================================================================
#
#==============================================================================

# lamda[i, j] : p(zj|di)
lamda = random([N, K])
for i in range(0, N):
    normalization = sum(lamda[i, :])
    for j in range(0, K):
        lamda[i, j] /= normalization;

# theta[i, j] : p(wj|zi)
theta = random([K, M])
for i in range(0, K):
    normalization = sum(theta[i, :])
    for j in range(0, M):
        theta[i, j] /= normalization;

#==============================================================================
#
#==============================================================================

# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K])

#==============================================================================
# E-Step
#==============================================================================
def EStep():
    for i in range(0, N):
        for j in range(0, M):
            denominator = 0;
            for k in range(0, K):
                p[i, j, k] = theta[k, j] * lamda[i, k];
                denominator += p[i, j, k];
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0;
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator;


#==============================================================================
# M-Step
#==============================================================================
def MStep():
    
    for k in range(0, K):
        denominator = 0
        for j in range(0, M):
            theta[k, j] = 0
            for i in range(0, N):
                theta[k, j] += X[i, j] * p[i, j, k]
            denominator += theta[k, j]
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
        

   
    for i in range(0, N):
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                lamda[i, k] += X[i, j] * p[i, j, k]
                denominator += X[i, j];
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator

def LogLikelihood():
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            tmp = 0
            for k in range(0, K):
                tmp += theta[k, j] * lamda[i, k]
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    print('loglikelihood : ', loglikelihood)

#==============================================================================
# EM algorithm
#==============================================================================
LogLikelihood()
for i in range(0, 20):
    EStep()
    MStep()
    print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] After the", i+1, "'s iteration  ", )
    LogLikelihood()


#==============================================================================
# get top words of each topic
#==============================================================================
topicwords = []
maxTopicWordsNum = 10
for i in range(0, K):
    topicword = []
    ids = theta[i, :].argsort()
    for j in ids:
        topicword.insert(0, dictionaryReverse[j])
    topicwords.append(topicword[0:min(maxTopicWordsNum, len(topicword))])

