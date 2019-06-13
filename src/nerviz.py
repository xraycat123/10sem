import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.tokenizer import Tokenizer
from IPython.core.display import display, HTML
from spacy.strings import StringStore
class NerVisualizer():

    def __init__(self, sentence,  palette = ['orange','pink','yellow','purple','grey']):
        self.sentence = sentence
        self.nlp = spacy.blank('en')
        self.groups = list(set([i.tag for i in sentence.get_spans('ner')] ))
        #self.groups = ["DIS"]
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        self._palette =  palette
        self.options = self.make_options()
        
    
    @property
    def palette(self):
        return self._palette
    
    @palette.setter
    def palette(self, colors):
        
        if len(colors)<len(self.groups):
            raise ValueError("You need more colors!")
        self._palette = colors
        self.options = self.make_options()
        self.plt()
    
    

    def make_options(self):
        colors ={}
        for k,group in enumerate(self.groups):
            colors[group] = self._palette[k]   
        return( {"ents": self.groups, "colors":  colors})
    
    def plt(self,jupyter = True):
        self.add_groups()
        one_sentence =  self.nlp(' '.join(self.sentence.to_tokenized_string().split()))
        
        for span in self.sentence.get_spans('ner'): 
            idx = [token.idx for token in span.tokens]
            #print(span.tag)
            span = Span(one_sentence, idx[0]-1, idx[-1], label=self.d[span.tag])
            #span = Span(one_sentence, idx[0]-1, idx[-1], label=self.d["DIS"])  # Create a span in Spacy
            one_sentence.ents = list(one_sentence.ents) + [span]  # add span to doc.ents 
        if jupyter == True: 
            display(HTML(spacy.displacy.render(one_sentence, style='ent',options=self.options))) # Use render in notebooks, or serve otherwise   
            self.html = (spacy.displacy.render(one_sentence, style='ent',options=self.options))
            
        else:
            spacy.displacy.serve(one_sentence, style='ent',options=self.options)
          
        
    def plt2(self,sent,tokx, jupyter = True):
        self.add_groups()
        one_sentence =  self.nlp(' '.join(sent.split()))
        
        for i in tokx: 
            idx = i
            #print(span.tag)
           # span = Span(one_sentence, idx[0]-1, idx[-1], label=self.d[span.tag])
            span = Span(one_sentence, idx[0]-1, idx[-1], label=self.d["DIS"])  # Create a span in Spacy
            one_sentence.ents = list(one_sentence.ents) + [span]  # add span to doc.ents 
        if jupyter == True: 
            display(HTML(spacy.displacy.render(one_sentence, style='ent',options=self.options))) # Use render in notebooks, or serve otherwise   
            self.html = (spacy.displacy.render(one_sentence, style='ent',options=self.options))
            
        else:
            spacy.displacy.serve(one_sentence, style='ent',options=self.options)
            
    
            
    def add_groups(self):
        self.d = {}
        for group in self.groups:
            stringstore = StringStore('u'+group)  # Change DIS to ORG, PER etc.
            disorder_hash = stringstore[group]
            self.d[group] = disorder_hash
            self.nlp.vocab.strings.add(group)

        
    
    def __call__(self):
        self.plt()
        
    def __repr__(self):
        return("Groups: " +  str(self.groups))