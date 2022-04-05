from os.path import join, exists
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class MLPipeline:
    def __init__(self, args, logger, k):
        self.data_path = args.data_path
        self.logger = logger
        self.k = k
        self.file_path = join(self.data_path, str(self.k))

        self.train_df = read_csv(join(self.file_path, 'train.txt'), header = None)
        self.val_df = read_csv(join(self.file_path, 'val.txt'), header = None)

        # Name columns.
        self.train_df.columns = self.val_df.columns = ('id', 'text', 'pol')

        # Vectorizer.
        error_msg = "\nThe vectorizer should be one of ['count', 'tfidf'].\n"
        assert (args.vectorizer.mode == "count") or (args.vectorizer.mode == "tfidf"), error_msg
        
        if args.vectorizer.mode == "count":
            # For text.
            self.vectorizer = CountVectorizer(lowercase = False,
                                        ngram_range = (1,2),
                                        max_features= args.vectorizer.max_features,
                                        preprocessor = lambda x: x,
                                        tokenizer = lambda sentence : self._tokenize(sentence),
                                        encoding = "utf-8")
        
        elif args.vectorizer.mode == "tfidf":
            # For text.
            self.vectorizer = TfidfVectorizer(lowercase = False,
                                        ngram_range = (1,2),
                                        max_features= args.vectorizer.max_features,
                                        preprocessor = lambda x: x,
                                        tokenizer = lambda sentence : self._tokenize(sentence),
                                        encoding = "utf-8")
    
    def _tokenize(self, string: str):
        return string.split()

    def _load_train_val(self):
        '''Load train and val data.'''
        return self.train_df, self.val_df


    def load_vectors(self):
        self.vectorizer.fit(self.train_df['text'])

        self.logger.info(f"\nNumber of 'text' features loaded: {len(self.vectorizer.get_feature_names_out())}")

        # Input features : Vectorized texts.
        x_train = self.vectorizer.transform(self.train_df['text'])
        x_val = self.vectorizer.transform(self.val_df['text'])
        
        # Gold Labels.
        y_train = self.train_df['pol']
        y_val = self.val_df['pol']
    
        self.logger.info(f'Train set shape : {x_train.shape}')
        self.logger.info(f'Val set shape : {x_val.shape}')

        return x_train, x_val, y_train, y_val

    def fit_predict(self, model):
        x_train, x_val, y_train, y_val = self.load_vectors()
        model.fit(x_train, y_train)
        # Predictions.
        y_pred_train = model.predict(x_train)
        y_pred_val = model.predict(x_val)

        return y_pred_train, y_pred_val


    
