from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import SequenceTaggingDataset

class Corpus(object):

  def __init__(self, input_folder, min_word_freq, batch_size):
    # list all the fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)

    SequenceTaggingDataset(, )
    # create dataset using built-in parser from torchtext
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=input_folder,
        train="train.tsv",
        validation="val.tsv",
        fields=(("word", self.word_field), ("tag", self.tag_field))
    )

    # convert fields to vocabulary list
    self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)

    # create iterator for batch input
    self.train_iter, self.val_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset),
        batch_size=batch_size
    )
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
