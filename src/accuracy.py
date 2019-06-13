from sklearn.metrics import confusion_matrix
from columnar import columnar
import copy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import numpy as np
import matplotlib.pyplot as plt
class Accuracy():
    def __init__(self, tagger: SequenceTagger, dataset: list):
        '''
         dataset is corpus.test, corpus.train etc
        '''
        self.tagger = tagger
        self.dataset = dataset
        self.trainer = ModelTrainer(tagger, dataset)
        self.eval_ds = self.trainer.evaluate(self.tagger, self.dataset)
    def get_acc(self,class_:str):
        
        print("accuracy for ", class_)
        print("TP: ",self.eval_ds[0].get_tp(class_))
        print("TN:",self.eval_ds[0].get_tn(class_))
        print("FP:",self.eval_ds[0].get_fp(class_))
        print("FN:",self.eval_ds[0].get_fn(class_))
    
    def sen_len(self):
        print(len(self.dataset))
        
    
    def get_preds_trues(self, mes_err = True, print_table = False):
        preds = []
        trues = []
        error = []
        error_table = []

        for sent_i in range(len(self.dataset)):
            index = sent_i
            true: Sentence = self.dataset[index]
            predi: Sentence = copy.deepcopy(self.dataset[index])
            self. tagger.predict(predi)
            lock = False
            #mes_err = True
            for i in range(len(predi.tokens)):
                pred_val = predi.tokens[i].get_tag('ner').value
                true_val = true.tokens[i].get_tag('ner').value

                preds.append(pred_val)
                trues.append(true_val)

                if mes_err == True: 
                    if pred_val != true_val and lock==False:
                        error.append((true,predi))
                        lock=True
           
        headers = ['True', 'Preds', 'Word']


        if mes_err == True:
          # data = []
            for f,j in error:
                data = []
                for i in range(len(f.tokens)):
                    row = [f.tokens[i].get_tag('ner').value, j.tokens[i].get_tag('ner').value,str(f.tokens[i].text)]
                  #  print(row,j.tokens[i].text)
                    data.append(row)
                table = columnar(data, headers, no_borders=False)
                error_table.append(data)
                if print_table:
                    print(table)
                    print(f)
                    print("\n\n")
        
        self.trues = trues
        self.preds = preds
        self.error_table = error_table  # alle værdier
   #     self.error = error
        return trues, preds
        
        
        
    def plot_confusion_matrix(self,y_true=None, y_pred=None, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if y_true==None or y_pred==None: 
            if not hasattr(self, 'trues'): 
                y_true,y_pred= self.get_preds_trues(mes_err = False)
            else:
                y_true,y_pred = self.trues, self.preds

        if classes == None: 
            counts = Counter(y_true)
            vocab = sorted(counts, key=counts.get, reverse=True)
            vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
            truez = [vocab_to_int[i] for i in y_true]
            predz = [vocab_to_int[i] for i in y_pred]
            classes = vocab
            y_true = truez
            y_pred = predz
                        



        if not title:
            if normalize:
                title = 'Normaliseret confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
       
        print(type(cm))
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm *= 100.0
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        print(cm)
        classes = ['O','I-DIS','B-DIS']

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='Sand label',
               xlabel='Prædikteret label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.set_size_inches(6, 6)
        fig.savefig('test2png.svg')
        return ax


    def __repr__(self):
        return str(self.tagger.modules)
        
        
        
    


    