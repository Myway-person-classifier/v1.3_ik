import torch 
from torch.utils.data import Dataset
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, args, 
                 df,
                 tokenizer, 
                 max_length=512, 
                 is_train=False,
                 is_submission=False):
        self.args = args
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.is_submission = is_submission
        #print("is submission: ", is_submission)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        cur_line = self.df.iloc[idx]
        if self.is_submission:
            output = self._getitem_test(cur_line)
        else:
            if self.is_train:
                output = self._getitem_train(cur_line)
            else:
                output = self._getitem_val(cur_line, idx)
        return output
        
    def _getitem_train(self, cur_line): #title, full_text, generated
        
        title = cur_line['title']  #str
        full_text = str(cur_line['full_text']) #str
        label = cur_line['generated'] #int
        
        paragraph_text = full_text.split('\n') #list of str
        paragraph_index = [i for i in range(len(paragraph_text))]
        
        item = {
            "title": title,
            'full_text': full_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
            "label": label            
        }        
        
        return item

    def _getitem_val(self, cur_line, idx): #title, full_text, generated
        
        title = cur_line['title']  #str
        full_text = str(cur_line['full_text']) #str
        label = cur_line['generated'] #int
        
        paragraph_text = full_text.split('\n') #list of str
        paragraph_index = [i for i in range(len(paragraph_text))]
        
        item = {
            "title": title,
            'full_text': full_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
            "label": label,
            'idx': idx            
        }        
        
        return item
    
    def _getitem_test(self, cur_line): #ID, title, paragraph_index, paragraph_text

        ID = cur_line['ID']
        title = cur_line['title']
        paragraph_index = cur_line['paragraph_index']  
        paragraph_text = cur_line['paragraph_text']

        item = {
            "ID": ID,
            "title": title,
            'full_text': paragraph_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
        }

        return item