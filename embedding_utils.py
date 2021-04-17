import pandas as pd
import re
import nltk
#from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from time import time
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE

INPUT_DATA_FILE = "data/train.csv" 
CLEANED_TEXT_FILE = "embedding/cleaned_text.csv"
MODEL_FILE = 'embedding/word2vec.model'
WORD_VECTORS_FILE = 'embedding/word2vec.wordvectors'
DOC_FILE = "data/full_document.txt"

# Stopwords to omit from vocabulary  
# using from nltk.corpus import stopwords caused performance issues
stopwords = ["date", "birth", "admission", "discharge", "sex", "age", "service", 
		     "allergies", "patient", "attending", "name","myself", "our", "ours", 
		     "ourselves", "you", "your", "yours", "yourself", "yourselves", "him", 
		     "his", "himself", "she", "her", "hers", "herself", "its", "itself", 
		     "they", "them", "their", "theirs", "themselves", "what", "which", "who", 
		     "whom", "this", "that", "these", "those", "are", "was", "were", "been", 
		     "being", "have", "has", "had", "having", "does", "did", "doing", "the", 
		     "and", "but", "because", "until", "while", "for", "with", "about", "against", 
		     "between", "into", "through", "during", "before", "after", "above", "below", 
		     "from", "down", "out", "off", "over", "under", "again", "further", "then", 
		     "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", 
		     "each", "few", "more", "most", "other", "some", "such", "nor", "not", "only", 
		     "own", "same", "than", "too", "very", "can", "will", "just", "should", "now"]


def load_data(data_file=INPUT_DATA_FILE):
	'''
		Loads data from a csv file. This is inteded
		 to load the MIMIC-III data and return the
		 text portion.

		Args:		data_file - The csv file to read in.
		Returns:	The Text data contained in the data file.
	'''

	text = pd.read_csv(data_file, usecols=['TEXT'])
	return text.TEXT

def clean_data(input_text, output_file=CLEANED_TEXT_FILE):

	'''
		Cleans and prepares the text to generate a Word2Vec model.

		The MIMIC-III data replaces protected health information like
		names, dates, etc with string patterns such as "Mr. [**Known lastname **]." 
		The resulting text pattern is littered throughout the date. 

		These string patterns are removed along with numbers and whitespace 
		characters.

		Args:		input_text - The input text data to be cleaned.
					ouput_file - The file path to save the cleaned data.
		Returns:	The cleaned data.
	'''

	for index, string in enumerate(input_text):

		# Initial cleaning of the text
		cleaned_text = string.lower()
		cleaned_text = re.sub('(?<=\[)(.*?)(?=\])', ' ', cleaned_text)
		cleaned_text = re.sub('[^a-zA-Z]', ' ', cleaned_text)
		cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

		# Each index is now a cleaned list of words from the original discharge notes
		input_text[index] = [word for word in cleaned_text.split(' ') if len(word) > 2]
		
		# Remove stop words
		input_text[index] = [word for word in input_text[index] if word not in stopwords]
	
	# Save the cleaned text for future use
	cleaned_series = pd.Series(input_text)
	cleaned_series.to_csv(output_file)	
	return input_text

def save_data(text_data, file=CLEANED_TEXT_FILE):
	'''
		Saves the text data to the specified path.

		Args: text_data - The data to be saved.
			  file 		- The file path to where the data are to be saved.
		Returns: N/A
	'''
	
	cleaned_series = pd.Series(text_data)
	cleaned_series.to_csv(file)	

def build_model(text_data, model_file=MODEL_FILE):
	'''
		Builds and saves a Word2Vec model from the input data.

		Args: 	 text_data -  The input data from which to build the model.
				 model_file - The path and filename of the saved model.
		Returns: The constructed model.
	'''

	model = Word2Vec(sentences=text_data, vector_size=100, window=5, min_count=3, workers=4)
	model.save(model_file)
	return model

def load_model(model=MODEL_FILE):
	'''
		Loads a Word2Vec model from a file.
		Args:		model - The model file to load.
		Returns:	The loaded model.
	'''

	return Word2Vec.load(model)

def train_model(model, data):
	'''
		Trains the model against a specified data set.

		Args:	model - The model to be trained.
				data  - The data against which the model is trained.
	'''

	model.train(data, total_examples=model.corpus_count, epochs=30, report_delay=1)

def save_embedding(model, file=WORD_VECTORS_FILE):
	'''
		Saves the word embedding from the trained model.

		Args:	model - The trained model.
				file  - The file path to the embedding.
		Returns: N/A
	'''

	model.wv.save(file)

def load_embedding(file=WORD_VECTORS_FILE):
	'''
		Loads a previously-saved embedding.

		Args:		file - the file path the saved embedding.
		Returns:	The loaded embedding.
	'''

	return KeyedVectors.load(file, mmap='r')

def print_vocab(model):
	'''
		Prints the model's vocabulary size and words.

		Args:		model - The model whose vocabulary is to be printed.
		Returns:	N/A
	'''

	print('Vocab length = %d' % len(model.wv))
	print(model.wv.key_to_index.keys())

def show_plot(word_vec):
	'''
		Show a plot of the embedding.

		Args:		word_vec - The vector of word embedding
		Returns:	N/A
	'''

	w = word_vec.key_to_index.keys()
	x = word_vec[w]
	w = list(w)
	x = x[:1000]
	w = w[:1000]
	y = TSNE().fit_transform(x)

	fig, ax = plt.subplots(figsize=(15, 15))
	ax.plot(y[:, 0], y[:, 1], 'o')
	ax.set_yticklabels([])
	ax.set_xticklabels([])

	for i, word in enumerate(w):
		if random.uniform(0,1) > 0.7:
			plt.annotate(word, xy=(y[i, 0], y[i, 1]))
	plt.show()

if __name__ == '__main__':
	'''
		Calling this script runs the end-to-end process to train an embedding.

		The following steps are taken:
			1) Loads the MIMIC-III data from a csv file and extracts the text
			2) Cleans the text data to prepare for model training and saves it
			3) Builds and saves a Word2Vec model from the cleaned data
			4) Trains the Word2Vec model to create and save an embedding

	'''

	print('Loading data')
	text_data = load_data()
	
	t = time()
	print('Data loaded and starting to clean text...')
	text_data = clean_data(text_data)
	print('Cleaning time: {} mins'.format(round((time() - t) / 60, 2)))

	print('Starting to build model...')
	t = time() 
	model = build_model(text_data)
	print('Model building time: {} mins'.format(round((time() - t) / 60, 2)))

	print('Training model...')
	t = time()
	train_model(model, text_data)
	print('Training time: {} mins'.format(round((time() - t) / 60, 2)))

	# Store the words and their trained embeddings
	save_embedding(model)
	print('Completed...')


	