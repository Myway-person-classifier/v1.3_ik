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
        
    def _getitem_train(self, cur_line):
        """
        Training 데이터 처리
        title이 없으면 빈 문자열 사용
        """
        # title이 있으면 사용, 없으면 빈 문자열
        title = cur_line.get('title', '') if 'title' in cur_line else ''
        full_text = str(cur_line['full_text'])  # str
        
        # ✅ 라벨 dtype 보정 (Suspect 1 대응)
        try:
            label = int(cur_line['generated'])
        except (ValueError, TypeError):
            # float인 경우 반올림 처리, 그 외에는 0/1 임계값 처리
            val = float(cur_line['generated'])
            label = 1 if val > 0.5 else 0
        
        paragraph_text = full_text.split('\n')  # list of str
        paragraph_index = [i for i in range(len(paragraph_text))]
        
        item = {
            "title": title,
            'full_text': full_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
            "label": label            
        }        
        
        return item

    def _getitem_val(self, cur_line, idx):
        """
        Validation 데이터 처리
        title이 없으면 빈 문자열 사용
        """
        # title이 있으면 사용, 없으면 빈 문자열
        title = cur_line.get('title', '') if 'title' in cur_line else ''
        full_text = str(cur_line['full_text'])  # str
        
        # ✅ 라벨 dtype 보정 (Suspect 1 대응)
        try:
            label = int(cur_line['generated'])
        except (ValueError, TypeError):
            val = float(cur_line['generated'])
            label = 1 if val > 0.5 else 0
        
        paragraph_text = full_text.split('\n')  # list of str
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
    
    def _getitem_test(self, cur_line):
        """
        Test 데이터 처리 (submission)
        """
        # Test 데이터는 다양한 형식 지원
        ID = cur_line.get('ID', cur_line.get('id', idx))
        title = cur_line.get('title', '') if 'title' in cur_line else ''
        
        # paragraph_text가 이미 리스트인 경우와 문자열인 경우 모두 처리
        if 'paragraph_text' in cur_line:
            paragraph_text = cur_line['paragraph_text']
            if isinstance(paragraph_text, str):
                paragraph_text = paragraph_text.split('\n')
        elif 'full_text' in cur_line:
            paragraph_text = str(cur_line['full_text']).split('\n')
        else:
            paragraph_text = ['']
        
        paragraph_index = cur_line.get('paragraph_index', list(range(len(paragraph_text))))

        item = {
            "ID": ID,
            "title": title,
            'full_text': '\n'.join(paragraph_text) if isinstance(paragraph_text, list) else paragraph_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
        }

        return item

