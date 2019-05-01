import pandas as pd
import numpy as np
import re
import collections
from scipy import stats

from spacy.lang.en.stop_words import STOP_WORDS
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
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import string

# descriptive analysis
# lable part
label_df = pd.read_csv("tweet_by_ID_18_3_2019__04_21_47.txt.labels",header=None)
df_count = pd.DataFrame()
df_count['count'] = label_df[0].value_counts()
df_count['percentage'] = df_count['count']/sum(df_count['count'].tolist())
df_percent = df_count.copy()
df_percent.drop(['count'], axis=1)
df_percent.to_csv('label_distribution.csv',index=False)

# training tweet
f = open("tweet_by_ID_18_3_2019__04_21_47.txt").read()
f_list = f.split('\n')
len(f_list)
text_df = pd.DataFrame()
text_df[0] = f_list

def clean_text(text_df):
	text_df.drop(text_df.tail(1).index,inplace=True)
	most_frequent = text_df.copy()
	# nltk Tweet tokenizer: https://www.nltk.org/api/nltk.tokenize.html
	# tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	tweet_tokenizer = RegexpTokenizer(r'\w+')
	most_frequent['text_list'] = most_frequent[0].apply(lambda x:tweet_tokenizer.tokenize(x))

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
# add targets into dataframe
ready_df['target'] = label_df[0]
train_df = ready_df[['text_string','target']]
print('train df shape')
print(train_df.shape)

# test label
f_test_label = open("us_test.labels").read()
f_test_label_list = f_test_label.split('\n')

# test text
f_test = open("us_test.text").read()
f_test_list = f_test.split('\n')
test_text_df = pd.DataFrame()
test_text_df[0] = f_test_list

test_ready_df = clean_text(test_text_df)
test_ready_df['target'] = f_test_label_list[:-1]
print('test df shape')
print(test_ready_df.shape)
test_df = test_ready_df[['text_string','target']]

# pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(sublinear_tf=True)),('svm', OneVsRestClassifier(LinearSVC(C=0.1, random_state=0))),])
print('get pipeline')
print('training')
text_clf.fit(train_df.text_string, train_df.target)
predicted = text_clf.predict(test_df.text_string)
test_df['predicted'] = predicted
test_df['predicted'] = test_df['predicted'].apply(lambda x:str(x))

# np.mean(test_df['predicted'] == test_df.target)
tfidf_cm = metrics.confusion_matrix(test_df['target'], test_df['predicted'])
mat = np.matrix(tfidf_cm)
mat.dump("unigram_matrix.dat")
# mat2 = numpy.load("my_matrix.dat")

y_true = test_df['target']
y_pred = test_df['predicted']
print('precision recall f1macro')
print(precision_recall_fscore_support(y_true, y_pred, average='macro'))
print('accuracy')
print(accuracy_score(y_true, y_pred))

# print(f1_score(test_df['predicted'], test_df['target'], average='macro'))
# print(f1_score(test_df['predicted'], test_df['target'], average='micro'))

test_df_gold = test_df[['target']]
test_df_pred = test_df[['predicted']]
test_df_gold.to_csv('unigram_gold.csv',index=False)
test_df_pred.to_csv('unigram_pred.csv',index=False)
# official score
# https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/tree/master/tools/evaluation%20script


