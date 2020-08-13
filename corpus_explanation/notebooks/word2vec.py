import re
import numpy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from numpy import *
from gensim.test.utils import datapath
import gensim.downloader as api
DATA_PATH = "../.data/imdb/aclImdb/train/pos/pos.txt"
with open(DATA_PATH) as file:
    text_review = file.read()

model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
word_vectors = model.wv
#if you want to use Google original vectors from Google News corpora
#model = KeyedVectors.load_word2vec_format('googlenewswv.bin.gz', binary=True)
#if you want to use your own vector
#model = word2vec.Word2Vec.load("w2v")

def text_to_wordlist(text, remove_stopwords=True):
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", text)

    # 3. Convert words to lower case and split them, clean stopwords from model' vocabulary
    words = review_text.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    return (meaningful_words)


# Function to get feature vec of words
def get_feature_vec(words, model):
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed 
    #index2word_set = set(model.index2wordword2vec.Word2Vec.load_word2vec_format)
    index2word_set = model.wv
    clean_text = []
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            clean_text.append(model[word])

    return clean_text

print("Cleaning text...")
# bag of word list without stopwords
clean_train_text = (text_to_wordlist(text_review, remove_stopwords=True))

# delete words which occur more than ones
clean_train = []
for words in clean_train_text:
    if words in clean_train:
        words = +1
    else:
        clean_train.append(words)
print("Getting feature vectors")
trainDataVecs = get_feature_vec(clean_train, model)
trainData = numpy.asarray(trainDataVecs)

# calculate cosine similarity matrix to use in pagerank algorithm for dense matrix, it is not
# fast for sparse matrix
# sim_matrix = 1-pairwise_distances(trainData, metric="cosine")

# similarity matrix, it is 30 times faster for sparse matrix
# replace this with A.dot(A.T).todense() for sparse representation
similarity = numpy.dot(trainData, trainData.T)

# squared magnitude of preference vectors (number of occurrences)
square_mag = numpy.diag(similarity)

# inverse squared magnitude
inv_square_mag = 1 / square_mag

# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[numpy.isinf(inv_square_mag)] = 0

# inverse of the magnitude
inv_mag = numpy.sqrt(inv_square_mag)

# cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag


# pagerank powermethod
def powerMethod(A, x0, m, iter):
    n = A.shape[1]
    delta = m * (array([1] * n, dtype='float64') / n)
    for i in range(iter):
        x0 = dot((1 - m), dot(A, x0)) + delta
    return x0


n = cosine.shape[1]  # A is n x n
m = 0.15
x0 = [1] * n
print("PageRank...")
pagerank_values = powerMethod(cosine, x0, m, 130)

srt = numpy.argsort(-pagerank_values)
a = srt[0:300]
print("Printing wordlist")
keywords_list = []

for words in a:
    keywords_list.append(clean_train_text[words])
    
print(keywords_list)

vocab = []
i = 1
while i <len(a):
	phrase = a[i-1]
	c_word = a[i]
	while f"{phrase} {c_word}" in text_review and i+1 < len(a):
		phrase = f"{phrase} {c_word}"
		i+=1
		c_word = a[i]
        vocab +=phrase
        i=i+1

print(vocab)


