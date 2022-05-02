import os
from typing import Dict, Tuple
from numpy import ones
from torch import as_tensor, index_put_
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import SequenceTaggingDataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import BertTokenizer

from utilities.utils import flatten, read_CoNLL

class AspectExtractionCorpus:

	def __init__(self, model, input_directory, device, model_name):
		
		self.input_directory = input_directory
		self.device = device
		self.model = model

		# Define the fields.
		self.word_field = Field(batch_first = True)
		self.tag_field = Field(unk_token = None, batch_first = True)
		self.FIELDS = (("word", self.word_field), ("tag", self.tag_field))

		# Create train and validation dataset using built-in parser from torchtext.
		self.train_ds, self.val_ds = SequenceTaggingDataset.splits(path = input_directory, 
																	train = 'train.txt', 
																	validation = 'val.txt', 
																	fields = self.FIELDS)

		# Convert fields to vocabulary list.
		self.word_field.build_vocab(self.train_ds.word)   # ADD VECTORS HERE.
		self.tag_field.build_vocab(self.train_ds.tag)

		# Prepare padding index to be ignored during model training/evaluation.
		self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
		self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]

		# Vocabulary and Tagset size.
		self.vocab_size = len(self.word_field.vocab.itos)	# Includeds <pad> and <unk> as well.
		self.tagset_size = len(self.tag_field.vocab.itos)	# Includes <pad> as well.

	def print_statistics(self):
		"""
		Prints the data statistics.
		"""
		print('\nLocation of dataset : ', self.input_directory)
		print('Length of training dataset : ', len(self.train_ds))
		print('Length of validation dataset : ', len(self.val_ds))
		print('Length of text vocab (unique words in dataset) : ', self.vocab_size)
		print('Length of label vocab (unique tags in labels) : ', self.tagset_size)
		print()

	def load_data(self, batch_size: int, shuffle: bool = False):
		'''
		Generates the data iterators for train and validation data.

		Parameters
		----------
		batch_size : int
			batch_size.
		shuffle : Bool, optional
			Whether to shuffle the data before training/testing. The default is True.

		Returns
		-------
		train_iter : training Dataloader instance.
		val_iter : validation Dataloader instance.
		'''
		train_dl, val_dl = BucketIterator.splits(datasets = (self.train_ds, self.val_ds),
															batch_sizes = (batch_size, batch_size),
															shuffle = shuffle,
															sort_key = lambda x: len(x.word),
															sort_within_batch = True,
															repeat = False,
															device = self.device)

		return train_dl, val_dl													  


class AspectDataset(Dataset):
	def __init__(self, input_directory: str, tokenizer, max_len: int, labels_to_ids: Dict = None, is_train: bool = True):
		"""
		Dataset class for Transformer architecture.

		Args:
			input_directory (str): Directory containing "train.txt" and "val.txt".
			tokenizer (_type_): Tokenizer instance.
			max_len (int): Maximum length of the sequence.
			labels_to_ids (Dict, optional): Mapping from labels to integers. Self generated if the the data is train set. If the data is val set, you neeed to supply the labels mapping.
											Defaults to None.
			is_train (bool, optional): Whether the data is train set. Defaults to True.
		"""	  	
		self.input_directory = input_directory
		self.tokenizer = tokenizer
		self.max_len = max_len

		self.sentences, self.labels = read_CoNLL(self.input_directory)
		self.tags_list = set(flatten(self.labels))

		if is_train:
			self.labels_to_ids = {k: v for v, k in enumerate(self.tags_list)}
			self.ids_to_labels = {v: k for v, k in enumerate(self.tags_list)}
		else:
			self.labels_to_ids = labels_to_ids
			self.ids_to_labels = {v : k for k, v in self.labels_to_ids.items()}


	def __getitem__(self, index):

		# step 1: get the sentence and word labels 
		sentence = self.sentences[index]  
		word_labels = self.labels[index] 

		# step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
		# BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
		encoding = self.tokenizer(sentence,
								  is_split_into_words=True, 
								  return_offsets_mapping=True, 
								  padding='max_length', 
								  truncation=True, 
								  max_length=self.max_len)
		
		# step 3: create token labels only for first word pieces of each tokenized word
		labels = [self.labels_to_ids[label] for label in word_labels] 
		
		# code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
		# create an empty array of -100 of length max_length
		encoded_labels = ones(len(encoding["offset_mapping"]), dtype=int) * -100
		
		# set only labels whose first offset position is 0 and the second is not 0
		i = 0
		for idx, mapping in enumerate(encoding["offset_mapping"]):
			if mapping[0] == 0 and mapping[1] != 0:
				encoded_labels[idx] = labels[i]
				i += 1

		# step 4: turn everything into PyTorch tensors
		item = {key: as_tensor(val) for key, val in encoding.items()}
		item['labels'] = as_tensor(encoded_labels)

		# Get lengths.
		item['seq_length'] = sum(encoding['attention_mask'])
		
		return item

	def __len__(self):
		return len(self.sentences)


class AspectExtractionCorpus_Transformer:
	def __init__(self, input_directory, tokenizer, max_len) -> None:
		self.input_directory = input_directory
		self.max_len = max_len
		self.tokenizer = tokenizer

		self.train_ds = AspectDataset(os.path.join(self.input_directory, "train.txt"), tokenizer, max_len)
		self.labels_to_ids = self.train_ds.labels_to_ids
		self.val_ds = AspectDataset(os.path.join(self.input_directory, "val.txt"), self.tokenizer, self.max_len, self.labels_to_ids, is_train=False)

	def load_data(self, batch_sizes: Tuple[int], shuffle: bool = True):
		train_dl = DataLoader(self.train_ds, batch_size = batch_sizes[0], shuffle = shuffle)
		val_dl = DataLoader(self.train_ds, batch_size = batch_sizes[1], shuffle = shuffle)
		return train_dl, val_dl
