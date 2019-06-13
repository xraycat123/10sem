import gensim 
from pathlib import Path
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

class CreateWordVectors():
    ''' 
    Class that creates w2v and character embeddings
    
    Args:
        datafile (str): The path for the textcorpus used for creation of the embeddings
    
    '''
    def __init__(self, datafile):
        
        self.datafile = datafile
        
    
    def _w2vtxt_processer(self, inputfile):
        with open(inputfile, 'rb') as f:
            for i, line in enumerate (f): 
                yield gensim.utils.simple_preprocess (line)
                
    def __call__(self,**kwargs):
        
        if 'word2vec' in kwargs:
            params = kwargs['word2vec']
            documents = list (self._w2vtxt_processer(self.datafile))
            self.w2vmodel = gensim.models.Word2Vec (documents, size=params['size'],
                                            min_count=params['min_count'], negative=params['negative'])
            self.w2vmodel.train(documents,total_examples=len(documents),epochs=10)
            
        if 'charemb' in kwargs:
            params = kwargs['charemb']
            is_forward_lm = params['forward']
            dictionary = Dictionary.load('chars')
            corpus = TextCorpus(Path('C:/Users/Martin/corpus'),
                    dictionary,
                    is_forward_lm,
                    character_level=True)
            
            language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=params['hidden_size'],
                               nlayers=1)
            # train your language model
            trainer = LanguageModelTrainer(language_model, corpus)
            trainer.train('resources/taggers/lang',
              sequence_length=params['sequence_length'],
              mini_batch_size=10,
              max_epochs=10)
            