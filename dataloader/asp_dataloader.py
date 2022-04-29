from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import SequenceTaggingDataset

class AspectExtractionCorpus:

	def __init__(self, input_directory, device):
		
		self.input_directory = input_directory
		self.device = device

		# List all the fields.
		self.word_field = Field(lower=True, batch_first=True)
		self.tag_field = Field(unk_token=None, batch_first=True)
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
		self.vocab_size = len(self.word_field.vocab.itos)    # Includeds <pad> and <unk> as well.
		self.tagset_size = len(self.tag_field.vocab.itos)    # Includes <pad> as well.

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