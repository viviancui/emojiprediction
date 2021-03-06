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

from sklearn.preprocessing import FunctionTransformer

import gensim, logging
from gensim.models import word2vec
from gensim.models import Doc2Vec

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from nltk.tokenize import RegexpTokenizer

# descriptive analysis
# lable part
label_df = pd.read_csv("tweet_by_ID_18_3_2019__04_21_47.txt.labels",header=None)
df_count = pd.DataFrame()
df_count['count'] = label_df[0].value_counts()
df_count['percentage'] = df_count['count']/sum(df_count['count'].tolist())
df_percent = df_count.copy()
df_percent.drop(['count'], axis=1)
df_percent.to_csv('label_distribution.csv',index=False)

# ax = sns.barplot(df_percent.index, "percentage", data=df_percent,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2")
# ax

#######
# https://radimrehurek.com/gensim/scripts/glove2word2vec.html
# doc2vec_model = gensim.models.Doc2Vec.load('doc2vec_external.model')
from_glove = "./glove.twitter.27B.100d.txt"
word2vec_file = get_tmpfile("./glove_word2vec.txt")
_ = glove2word2vec(from_glove, word2vec_file)
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file)
print(word2vec_model['test'])


# word_vecs_matrix = get_wv_matrix()  # pseudo-code
def transform(x):
	# result = doc2vec_model.infer_vector(x)
	# return result
	embedding_list = []
	count = 0
	for ele in x:
		try:
			embedding_list.append(word2vec_model[ele])
			count += 1
		except:
			pass
			# embedding_list.append(np.zeros(100))
	if count != 0:
		results = np.array(embedding_list)
		result = np.mean(results,axis=0,dtype=np.float64)
	else:
		result = np.zeros(100,dtype=np.float64)
	return result
# transformer = FunctionTransformer(transform)
#######

# training tweet
f = open("tweet_by_ID_18_3_2019__04_21_47.txt").read()
f_list = f.split('\n')
len(f_list)
text_df = pd.DataFrame()
text_df[0] = f_list
text_df['target'] = label_df[0]

def clean_text(text_df):
	text_df.drop(text_df.tail(1).index,inplace=True)
	most_frequent = text_df.copy()
	# tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	tweet_tokenizer = RegexpTokenizer(r'\w+')

	most_frequent['text_list'] = most_frequent[0].apply(lambda x:tweet_tokenizer.tokenize(x))
	text_all_frequent = most_frequent['text_list'].tolist()
	text_all_frequent_flat = [item for sublist in text_all_frequent for item in sublist]
	text_all_frequent_count = collections.Counter(text_all_frequent_flat)
	most_frequent_100 = text_all_frequent_count.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_200_words = most_frequent_100_keys[:200]
	print(top_200_words)


	# remove punctuation
	import re
	def remove_punc(text_list):
	    for ele in text_list:
	        if re.match("\W+", ele):
	            text_list.remove(ele)
	    return text_list

	no_punc = most_frequent.copy()
	no_punc['text_list_no_punc'] = no_punc.text_list.apply(lambda x:remove_punc(x))
	no_punc.head()   

	text_all_frequent2 = no_punc['text_list'].tolist()
	text_all_frequent_flat2 = [item for sublist in text_all_frequent2 for item in sublist]
	text_all_frequent_count2 = collections.Counter(text_all_frequent_flat2)
	most_frequent_100 = text_all_frequent_count2.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_200_words_cleaned = most_frequent_100_keys[:200]
	print(top_200_words_cleaned)

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

	no_stop['len'] = no_stop['text_list_no_stop'].apply(lambda x:len(x))
	no_stop = no_stop[no_stop.len != 0]

	no_stop['doc2vec'] = no_stop['text_list_no_stop'].apply(lambda x:transform(x))
	# no_stop['len_vec'] = no_stop['word2vec'].apply(lambda x:x.size)
	# no_stop = no_stop[no_stop.len_vec != 0]
	# print('after clean shape')
	# print(no_stop.size)
	# no_stop = no_stop[no_stop.word2vec != np.nan]

	ready_df = no_stop.copy()
	ready_df['text_string'] = ready_df.text_list_no_stop.apply(lambda x:' '.join(x))
	ready_df.head()

	text_all_frequent_after = ready_df['text_list_no_stop'].tolist()
	text_all_frequent_flat_after = [item for sublist in text_all_frequent_after for item in sublist]
	text_all_frequent_count_after = collections.Counter(text_all_frequent_flat_after)

	most_frequent_100 = text_all_frequent_count_after.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	print(most_frequent_100_keys)

	return ready_df

ready_df = clean_text(text_df)

# add targets into dataframe
train_df = ready_df[['doc2vec','target']]



# test label
f_test_label = open("us_test.labels").read()
f_test_label_list = f_test_label.split('\n')

# test_df = pd.DataFrame()
# test_df['target']=f_test_label_list[:-1]

# test text
f_test = open("us_test.text").read()
f_test_list = f_test.split('\n')
test_text_df = pd.DataFrame()
test_text_df[0] = f_test_list
print(test_text_df.shape)
print(len(f_test_label_list))
test_text_df['target'] = f_test_label_list

test_text_df = test_text_df.head(50000)
test_ready_df = clean_text(test_text_df)
print(test_ready_df.shape)
test_df = test_ready_df[['doc2vec','target']]
test_df.to_csv('check_test_df.csv')

#######
# use embedding
# https://stackoverflow.com/questions/49236166/how-to-make-use-of-pre-trained-word-embeddings-when-training-a-model-in-sklearn

# this assumes you're using numpy ndarrays


# ###################
# text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer(sublinear_tf=True)),('svm', OneVsRestClassifier(SVC(C=0.1, random_state=0))),])
# text_clf = Pipeline([('tfidf', TfidfTransformer(sublinear_tf=True)),('svm', OneVsRestClassifier(SVC(C=0.1, random_state=0)))])
text_clf = OneVsRestClassifier(LinearSVC(C=0.1, random_state=0))
print('got pipeline')
print('start training')
X = train_df.doc2vec.tolist()
y = train_df.target
text_clf.fit(X, y)
predicted = text_clf.predict(test_df.doc2vec.tolist())
test_df['predicted'] = predicted
test_df['predicted'] = test_df['predicted'].apply(lambda x:str(x))

# np.mean(test_df['predicted'] == test_df.target)
tfidf_cm = metrics.confusion_matrix(test_df['target'], test_df['predicted'])
mat = np.matrix(tfidf_cm)
mat.dump("tfidf_matrix_doc2vec.dat")
# mat2 = numpy.load("my_matrix.dat")

# print(f1_score(test_df['predicted'], test_df['target'], average='macro'))
# print(f1_score(test_df['predicted'], test_df['target'], average='micro'))
test_df_gold = test_df[['target']]
test_df_pred = test_df[['predicted']]
test_df_gold.to_csv('glove_w2v_gold.csv',index=False)
test_df_pred.to_csv('glove_w2v_pred.csv',index=False)

test_df_new = test_df[['predicted','target']]
test_df_new.to_csv('glove_w2v_result.csv',index=False)
