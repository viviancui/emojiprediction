import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as Data
from torch.utils.data import Dataset, DataLoader

import re
import pandas as pd
import numpy as np
from scipy import stats
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

class bilstm(nn.Module):
	def __init__(self,embedding_dimension, hidden_dimension, vocab_size, target_set_size, batch_size, dropout):
		super(bilstm, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dimension, padding_idx=0)
		self.hidden_dimension = hidden_dimension
		self.batch_size = batch_size
		# self.dropout = dropout
		self.dropout = nn.Dropout(p=self.config.keep_prob)
		self.lstm = nn.LSTM(embedding_dimension,hidden_dimension,bidirectional=True)
		self.hidden_to_target = nn.Linear(hidden_dimension*2, target_set_size)
		self.hidden = self.init_hidden()

	def forward(self,input):
		embeds = self.embeddings(input).view(input.size()[1],self.batch_size, -1) # the dimension should correspoding to the dataloader
		output, self.hidden = self.lstm(embeds,self.hidden)
		predict = self.hidden_to_target(output[-1])
		predict = F.log_softmax(predict,dim=1) #,dim=1
		return predict

	def init_hidden(self):
		return torch.zeros(2,self.batch_size,self.hidden_dimension),torch.zeros(2,self.batch_size,self.hidden_dimension)



def remove_punc(text_list):
    for ele in text_list:
        if re.match("\W+", ele):
            text_list.remove(ele)
    return text_list

def list_lower(list):
	    new_list = [ele.lower() for ele in list]
	    return new_list

def clean_text(text_df):
	text_df.drop(text_df.tail(1).index,inplace=True)
	most_frequent = text_df.copy()
	# tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	tweet_tokenizer = RegexpTokenizer(r'\w+')
	most_frequent['text_list'] = most_frequent[0].apply(lambda x:tweet_tokenizer.tokenize(x))
	text_all_frequent = most_frequent['text_list'].tolist()
	text_all_frequent_flat = [item for sublist in text_all_frequent for item in sublist]
	text_all_frequent_count = Counter(text_all_frequent_flat)
	most_frequent_100 = text_all_frequent_count.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_200_words = most_frequent_100_keys[:200]
	# print(top_200_words)
	print('got top 200')

	# remove punctuation
	no_punc = most_frequent.copy()
	no_punc['text_list_no_punc'] = no_punc.text_list.apply(lambda x:remove_punc(x))
	no_punc.head()   

	text_all_frequent2 = no_punc['text_list'].tolist()
	text_all_frequent_flat2 = [item for sublist in text_all_frequent2 for item in sublist]
	text_all_frequent_count2 = Counter(text_all_frequent_flat2)
	most_frequent_100 = text_all_frequent_count2.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	top_200_words_cleaned = most_frequent_100_keys[:200]
	# print(top_200_words_cleaned)
	print('got top 200 cleaned')

	# change to lowercase
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
	text_all_frequent_count_after = Counter(text_all_frequent_flat_after)

	most_frequent_100 = text_all_frequent_count_after.most_common(100)
	most_frequent_100_keys = [x[0] for x in most_frequent_100]
	# print(most_frequent_100_keys)
	print('got most frequent 100')

	ready_df['len'] = ready_df.text_list_no_stop.apply(lambda x:len(x))
	lengths = ready_df.len.tolist()
	max_length = max(lengths)

	def add_padding(x):
		add_num = max_length - len(x)
		add_padding = ['<pad>']*add_num
		result = add_padding + x
		return result

	ready_df['text_list_no_stop'] = ready_df.text_list_no_stop.apply(lambda x:add_padding(x))

	return ready_df

def vocab_dictionary(train_df):
	total_list = train_df['text_list_no_stop']
	flatten_list = [item for sublist in total_list for item in sublist]
	unique_words = set(flatten_list)
	num_words = len(unique_words)
	word_to_index_dict = {word: i+1 for i,word in enumerate(unique_words)}
	word_to_index_dict[0] = '<pad>'
	return word_to_index_dict

class TweetsData(Dataset):
	def __init__(self):
		label_df = pd.read_csv("tweet_by_ID_18_3_2019__04_21_47.txt.labels",header=None)
		f = open("tweet_by_ID_18_3_2019__04_21_47.txt", 'r', encoding='utf-8').read()
		f_list = f.split('\n')
		len(f_list)
		text_df = pd.DataFrame()
		text_df[0] = f_list
		###
		# text_df = text_df.head(1000)
		###
		ready_df = clean_text(text_df)
		ready_df['target'] = label_df[0]
		word_to_index_dict = vocab_dictionary(ready_df)
		def convert_list_word_to_index(word_list):
			index_list = [word_to_index_dict[word] for word in word_list]
			return index_list
		ready_df['word_index'] = ready_df.text_list_no_stop.apply(lambda x:convert_list_word_to_index(x))
		train_df = ready_df[['word_index','target']]
		###
		# train_df = train_df.head(1000)
		###
		self.vocab_size = len(word_to_index_dict)
		print(len(word_to_index_dict))
		self.data = train_df

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		inputs = self.data.iloc[idx]['word_index']
		targets = self.data.iloc[idx]['target']
		inputs = torch.tensor(inputs)
		targets = torch.tensor(targets)
		return inputs,targets



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



batch_size = 2048
dataset = TweetsData()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,num_workers=1,drop_last=True)#
print('get train loader')


label_df = pd.read_csv("tweet_by_ID_18_3_2019__04_21_47.txt.labels",header=None)
###
# label_df = label_df.head(1000)
###
target_set_counter = Counter(label_df[0])
target_set_size = len(target_set_counter)
print(target_set_size)


hidden_dimension = 150
embedding_dimension = 300
model = bilstm(embedding_dimension=embedding_dimension, hidden_dimension=hidden_dimension, vocab_size=dataset.vocab_size, target_set_size=target_set_size, batch_size=batch_size, dropout=0.05)
model.to(device)
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



def train():
	for epoch in range(1):
		for i,data in enumerate(train_loader,0):
			inputs, targets = data
			inputs, targets = inputs.to(device), targets.to(device)
			model.zero_grad()
			tag_scores = model(inputs)
			loss = loss_function(tag_scores,targets)
			print(loss)
			loss.backward(retain_graph=True)
			optimizer.step()
	torch.save(model.state_dict(), 'model')



class TestData(Dataset):
	def __init__(self):
		f_test = open("us_test.text", 'r', encoding='utf-8').read()
		f_test_list = f_test.split('\n')
		print(len(f_test_list))
		test_df = pd.DataFrame()
		test_df[0] = f_test_list
		test_df.drop(test_df.tail(1).index,inplace=True)

		f_test_label = open("us_test.labels", 'r', encoding='utf-8').read()
		f_test_label_list = f_test_label.split('\n')
		print(len(f_test_label_list[:-1]))
		test_df['target']=f_test_label_list[:-1]

		ready_df = clean_text(test_df)
		# ready_df['target'] = label_df[0]
		word_to_index_dict = vocab_dictionary(ready_df)
		def convert_list_word_to_index(word_list):
			index_list = [word_to_index_dict[word] for word in word_list]
			return index_list
		ready_df['word_index'] = ready_df.text_list_no_stop.apply(lambda x:convert_list_word_to_index(x))
		train_df = ready_df[['word_index','target']]
		# train_df = train_df.head(1000)
		self.vocab_size = len(word_to_index_dict)
		print(len(word_to_index_dict))
		self.data = train_df

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		inputs = self.data.iloc[idx]['word_index']
		targets = self.data.iloc[idx]['target']
		inputs = torch.tensor(inputs)
		targets = targets
		return inputs,targets

test_dataset = TestData()
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=True)#
print('get test loader')

def test():
	# test_df.to_csv('predicted_output.csv')
	# calculate F1 accuracy
	model = bilstm(embedding_dimension=embedding_dimension, hidden_dimension=hidden_dimension, vocab_size=dataset.vocab_size, target_set_size=target_set_size, batch_size=1, dropout=0.05)
	# model.to(device)
	model.load_state_dict(torch.load('model'))
	targets_list = []
	predicted_list = []
	for i,data in enumerate(test_loader,0):
		inputs,targets = data
		# inputs, targets = inputs.to(device), targets.to(device)
		tag_scores = model(inputs)
		target_index = np.argmax(tag_scores.detach().numpy())
		tag_result = target_index
		targets_list.append(targets)
		predicted_list.append(tag_result)
	df_new = pd.DataFrame()
	df_new['targets'] = targets_list
	df_new['predicted'] = predicted_list
	df_new.to_csv('results_first_model.csv')

train()
test()



