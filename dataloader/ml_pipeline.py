import os
from os.path import join, exists
from pandas import DataFrame, read_csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

class MLPipeline:
    def __init__(self, args, logger, k):
        self.data_path = args.data_path
        self.logger = logger
        self.k = k
        self.file_path = join(self.data_path, str(self.k))

        self.train_df = read_csv(join(self.file_path, 'train.txt'), header = None)
        self.val_df = read_csv(join(self.file_path, 'val.txt'), header = None)

        # Name columns.
        self.train_df.columns = self.val_df.columns = ('text', 'pol')
        
        # Saved model directory.
        self.model_name = args.model
        self.save_model_dir = join(args.output_dir, self.model_name)

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

    def _fit_vectors(self):
        self.vectorizer.fit(self.train_df['text'])

    def load_vectors(self):
        self._fit_vectors()
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

        # Save model.
        self._save_model(model, filename = 'model.sav')

        return y_pred_train, y_pred_val

    def _raw_text_cleaner(self, text):
        return 

    def _save_model(self, model, filename = 'model.sav'):
        os.makedirs(self.save_model_dir, exist_ok = True)
        pickle.dump(model, open(join(self.save_model_dir, filename), 'wb'))

    def _load_model(self, filepath):
        '''
        Loads the saved model. Used for inference.
        '''
        loaded_model = pickle.load(open(filepath, 'rb'))
        return loaded_model

    def _write_results(self, text, predictions):
        '''
        Writes the results of the inference in a TSV file in the directory ./results/<model_name> .
        '''

        assert len(text) == len(predictions), f"The lengths of text and predictions don't match.\nText length : {len(text)}\nPredictions length : {len(predictions)}"
        os.makedirs(join('results', self.args.model), exist_ok = True)
        filename = f'results_{self.args.model}.tsv'
        df_res = DataFrame({'texts' : text, 'predictions' : predictions})
        df_res.to_csv(join('results', self.args.model, filename), sep = '\t', encoding = 'utf-8', index = None)

    def infer(self, model, x_test):
        x_test_cleaned = self._raw_text_cleaner(x_test)

        # Vectorize the cleaned text.
        x_test_vectorized = self.vectorizer.transform(x_test_cleaned)

        # Load the saved model.
        model = self._load_model(join(self.save_model_dir, 'model.sav'))

        # Create Predictions.
        y_pred_test = model.predict(x_test_vectorized)

        # Write results.
        self._write_results(x_test, y_pred_test)

        self.logger.info(f'\nResults of the test set saved as {join("results", self.args.model, f"results_{self.args.model}.tsv")}.')










    
