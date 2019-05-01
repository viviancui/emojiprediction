# from gensim.test.utils import common_texts, get_tmpfile
# from gensim.models import Word2Vec

# path = get_tmpfile("word2vec.model")
# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")
# # https://radimrehurek.com/gensim/models/word2vec.html

import pandas as pd
import numpy as np
import collections
from scipy import stats
from spacy.lang.en.stop_words import STOP_WORDS
# import seaborn as sns
# import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics

# from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import string

# descriptive analysis
# lable part
text_df = pd.read_csv("genuine_tweets.csv",low_memory=False)
text_df.dropna(subset=['text'],inplace=True)

def clean_text(text_df):
	text_df.drop(text_df.tail(1).index,inplace=True)
	most_frequent = text_df.copy()
	# nltk Tweet tokenizer: https://www.nltk.org/api/nltk.tokenize.html
	# tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	tweet_tokenizer = RegexpTokenizer(r'\w+')
	most_frequent['text_list'] = most_frequent['text'].apply(lambda x:tweet_tokenizer.tokenize(x))

	text_all_frequent = most_frequent['text_list'].tolist()
	text_all_frequent_flat = [item for sublist in text_all_frequent for item in sublist]
	text_all_frequent_count = collections.Counter(text_all_frequent_flat)
	most_frequent_100 = text_all_frequent_count.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_100_words = most_frequent_100_keys[:100]
	print('top 100 words before remove unrelavent things')
	print(top_100_words)

	# regular expression remove unrelated stuff, like punctuation, URLs etc.
	def remove_punc(text_list):
	    for ele in text_list:
	    	if ele in string.punctuation:
	    		text_list.remove(ele)
	    	elif ele == "@":
	    		text_list.remove(ele)
	    	elif ele == "!":
	    		text_list.remove(ele)
	    	elif ele == "-":
	    		text_list.remove(ele)
	    	elif ele == ".":
	    		text_list.remove(ele)
	    	elif ele == "_":
	    		text_list.remove(ele)
	    	elif ele == "•":
	    		text_list.remove(ele)
	    	elif ele == "…":
	    		text_list.remove(ele)
	    	elif ele == ":":
	    		text_list.remove(ele)
	    	elif ele == "・":
	    		text_list.remove(ele)
	    	elif ele == "(":
	    		text_list.remove(ele)
	    	elif ele == ")":
	    		text_list.remove(ele)
	    	elif ele == "“":
	    		text_list.remove(ele)
	    	elif ele == "#":
	    		text_list.remove(ele)
	    	elif ele == "...":
	    		text_list.remove(ele)
	    	elif ele == "..":
	    		text_list.remove(ele)
	    	elif ele == "'":
	    		text_list.remove(ele)
	    	elif ele == "/":
	    		text_list.remove(ele)
	    	elif ele == "”":
	    		text_list.remove(ele)
	    	elif ele == '"':
	    		text_list.remove(ele)
	    	elif ele == '’':
	    		text_list.remove(ele)
	    	else:
	    		pass
	    return text_list

	no_punc = most_frequent.copy()
	no_punc['text_list_no_punc'] = no_punc.text_list.apply(lambda x:remove_punc(x))

	text_all_frequent2 = no_punc['text_list_no_punc'].tolist()
	text_all_frequent_flat2 = [item for sublist in text_all_frequent2 for item in sublist]
	text_all_frequent_count2 = collections.Counter(text_all_frequent_flat2)
	most_frequent_100 = text_all_frequent_count2.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_100_words_cleaned = most_frequent_100_keys[:100]
	print('top 100 words after cleaning')
	print(top_100_words_cleaned)

	# change to lowercase
	def list_lower(list):
	    new_list = [ele.lower() for ele in list]
	    return new_list

	no_punc_lower = no_punc.copy()
	no_punc_lower['text_list_no_punc_lower'] = no_punc_lower.text_list_no_punc.apply(lambda x:list_lower(x))
	no_punc_lower.head()

	# remove stop words
	no_stop = no_punc_lower.copy()
	no_stop['text_list_no_stop'] = no_stop['text_list_no_punc_lower'].apply(lambda x: [i for i in x if i not in STOP_WORDS])
	no_stop.head()

	ready_df = no_stop.copy()
	ready_df['text_string'] = ready_df.text_list_no_stop.apply(lambda x:' '.join(x))
	ready_df.head()

	text_all_frequent_after = ready_df['text_list_no_stop'].tolist()
	text_all_frequent_flat_after = [item for sublist in text_all_frequent_after for item in sublist]
	text_all_frequent_count_after = collections.Counter(text_all_frequent_flat_after)

	most_frequent_100 = text_all_frequent_count_after.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	print('top 100 words without stop words')
	print(most_frequent_100_keys)

	return ready_df

ready_df = clean_text(text_df)
# print(ready_df.columns)
# add targets into dataframe
# ready_df['target'] = label_df[0]
train_df = ready_df[['text_list_no_stop']]


# train the model
import gensim, logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for line in self.dirname:
#             yield line.split()
            yield line

text_all = train_df['text_list_no_stop']
    
sentences = MySentences(text_all) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences,size=100, window=5, min_count=1, workers=1)
# model = word2vec.Word2Vec(sentences, workers=1)

# model.save('/tmp/mymodel')
model.save("external_word2vec.model")
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')



