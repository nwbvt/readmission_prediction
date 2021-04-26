import argparse
from torch.utils.data import DataLoader
import data
import model as md
import embedding_utils as em


def run(num_epochs, embedding=None):
	
	print("Loading embedding...")
	w2v = embedding if embedding is not None else em.load_embedding()
	vocab_size, embed_size = w2v.vectors.shape
	print("Finished loading embedding")

	print("Loading training data and creating training dataloader...")
	train_ds = data.load_dataset("data/train.csv", encoding=w2v.key_to_index.get)
	train_loader = DataLoader(train_ds, batch_size=10, collate_fn=data.collate, shuffle=True)
	print("Finished")

	print("Loading testing data and creating testing dataloader...")
	test_ds = data.load_dataset("data/test.csv", encoding=w2v.key_to_index.get)
	test_loader = DataLoader(test_ds, batch_size=10, collate_fn=data.collate, shuffle=False)
	print("Finished")

	print("Creating model...")
	model = md.ConvAttModel(vocab_size=vocab_size, embed_size=embed_size, embedding=w2v.vectors)
	print("Finished\nTraining model...")
	md.train(model, train_loader, test_loader, num_epochs)
	print("Finished")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Run Convolutional Model w/ Attention and Pre-trained Word Embedding")
	parser.add_argument("--epochs", "-e", help="The number of Epochs", default=1)
	args = parser.parse_args()
	run(int(args.epochs))