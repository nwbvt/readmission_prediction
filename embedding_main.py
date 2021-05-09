'''
	embedding_main.py

	This is a simple driver script to create and train word embeddings of 
	various dimension and window sizes so that they can be compared. 

	The values of DIMENSIONS, WINDOWS, and NUM_EPOCHS can be modified prior
	to running this script to avoid creating and training all combinations. 

	I modified model.py to print a single line to stdout and redirected stdout
	to a file to store the output of the attention model training. This script
	writes to stderr so that the pipeline status can be seen while the program
	runs without cluttering the results file.
'''

import os, sys
import argparse
from torch.utils.data import DataLoader
import data
import model as md
import embedding_utils as em
import pandas as pd

os.environ['PYTHONHASHSEED'] = '1313'

DIMENSIONS = ['50', '100', '250', '500']
WINDOWS = ['1', '5', '10']
NUM_EPOCHS = 5

def trainModel(num_epochs, embedding):
	'''
		Trains the attention model for the specified number of epochs
		using the specified embedding.

		Arguments:
			num_epochs: The number of training epochs
			embedding:  The embedding to use for training
	'''

	vocab_size, embed_size = embedding.vectors.shape
	
	train_ds = data.load_dataset("data/train.csv", encoding=embedding.key_to_index.get)
	train_loader = DataLoader(train_ds, batch_size=10, collate_fn=data.collate, shuffle=True)
	test_ds = data.load_dataset("data/test.csv", encoding=embedding.key_to_index.get)
	test_loader = DataLoader(test_ds, batch_size=10, collate_fn=data.collate, shuffle=False)
	
	model = md.ConvAttModel(vocab_size=vocab_size, embed_size=embed_size, embedding=embedding.vectors)
	md.train(model, train_loader, test_loader, num_epochs)

def createEmbeddings(dimensions=DIMENSIONS, windows=WINDOWS):
	'''
		Creates, trains, and saves word embeddings for the given dimension and window size combinations.
		The embedding model and word vector files will be saved in the embedding directory.

		Arugments:
			dimensions: A list of strings for the dimensionality of the word embedding
			windows: A list of strings for the window sizes for the embedding
	'''

	print('\nLoading data', file=sys.stderr)
	text_data = em.load_data(em.CLEANED_TEXT_FILE)

	for dim in dimensions:
		for win in windows:
			model_file = 'embedding/word2vec_{}d_{}win.model'.format(dim, win)
			embedding_file = 'embedding/word2vec_{}d_{}win.wordvectors'.format(dim, win)
			
			print('\nStarting pipeline for %s dimensions, %s window size' % (dim, win), file=sys.stderr)
			model = em.build_model(text_data, vector_size=int(dim), window=int(win))

			print('\nTraining embedding', file=sys.stderr)
			em.train_model(model, text_data)

			print('\nSaving model and embedding', file=sys.stderr)
			em.save_model(model, model_file)
			em.save_embedding(model, file=embedding_file)

def trainModelWithEmbeddings(dimensions=DIMENSIONS, windows=WINDOWS, epochs=NUM_EPOCHS):
	'''
		Trains the attention model for the given number of epochs for each
		word embedding combination of dimension and window sizes. The embeddings are
		expected to have been created and trained already and stored in the embedding
		directory.

		Arguments:
			dimensions: A list of strings for the embedding dimensions
			windows: A list of strings for the embedding window sizes
			epochs: The number of epoch to train the model
	'''

	for dim in dimensions:
		for win in windows:
			embedding_file = 'embedding/word2vec_{}d_{}win.wordvectors'.format(dim, win)
			print('\nTraining attention model with embedding for %s dimensions, %s window size' % (dim, win), file=sys.stderr)
			embedding = em.load_embedding(embedding_file)
			trainModel(epochs, embedding)

def trainWithBaselineEmbedding(enum_epochs):
	'''
		Helper function to train the attention model with 
		the baseline embedding for the number of epochs.

		Arguments:
			num_epochs 	- The number of epochs to train
	'''
	embedding = em.load_embedding('embedding/word2vec.wordvectors')
	trainModel(enum_epochs, embedding)
	
def trainWithBestEmbedding(enum_epochs):
	'''
		Helper function to train the attention model with 
		the top performing embedding for the number of epochs.

		Arguments:
			num_epochs	- The number of epochs to train
	'''
	print('\nTraining attention model with top performing word embedding.', file=sys.stderr)
	embedding = em.load_embedding('embedding/word2vec_250d_1win.wordvectors')
	trainModel(enum_epochs, embedding)
	
if __name__ == '__main__':
	
	# Note that this will create and train all 12 word embeddings
	# and train the attention model for 5 epochs each 
	createEmbeddings()
	trainModelWithEmbeddings()
