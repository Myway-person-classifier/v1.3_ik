import torch 
import torch.nn as nn

class TextCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        #train valid format
        """
        item = {
            "title": title,
            'full_text': full_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
            "label": label            
        }       
        """
        
        #subbmission format
        """
        item = {
            "title": title,
            'full_text': full_text,  # str
            "paragraph_index": paragraph_index,  # list of int
            "paragraph_text": paragraph_text,  # list of str
            "label": label            
        }        
        """        
        self.is_first_init = True        
        
    def __call__(self, batch):
        item = {}
        
        # labels 처리 (기존 코드)
        if 'paragraph' in self.args.save_dir:
            labels = (
                torch.tensor([x['label'] for x in batch], dtype=torch.float)
                if 'label' in batch[0] else None
            )
        else:
            labels = (
                torch.tensor([x['label'] for x in batch], dtype=torch.float)
                if 'label' in batch[0] else None
            )
        if labels is not None:
            item['labels'] = labels

        if self.args.split_valid_by_paragraph:
            paragraph_texts = [x['paragraph_text'] for x in batch]  # List[List[str]]
            flat_idx  = [x['idx'] for x in batch] # List[int]

            flat_paras = [p for paras in paragraph_texts for p in paras] #use flatten paras
            tok = self.tokenizer(
                flat_paras,
                padding='longest',
                truncation=True,
                max_length=self.args.max_length,
                return_tensors='pt'
            )

            #==================================================================================#
            # 2) 최소 길이 설정
            min_len = 5
            input_ids     = tok['input_ids']      # shape: [B, L]
            attention_mask= tok['attention_mask'] # shape: [B, L]

            cur_len = input_ids.size(1)
            if cur_len < min_len:
                pad_len   = min_len - cur_len
                # pad token id 로 채운 (B, pad_len) 텐서
                pad_ids   = input_ids.new_full((input_ids.size(0), pad_len),
                                            self.tokenizer.pad_token_id)
                pad_masks = attention_mask.new_zeros((attention_mask.size(0), pad_len))
                # 뒤쪽에 concat
                tok['input_ids']      = torch.cat([input_ids,     pad_ids],   dim=1)
                tok['attention_mask'] = torch.cat([attention_mask,pad_masks], dim=1)
            #==================================================================================#

            item['input_ids']      = tok['input_ids'] #[batch, seq_len]
            item['attention_mask'] = tok['attention_mask'] #[batch, seq_len]
            item['labels'] = torch.tensor(flat_idx, dtype=torch.float32)  # [batch, num_paras] 형태로 변환
            
            #print(f"Batch size: {len(batch)}, Paragraphs: {len(flat_paras)}")

        elif self.args.use_paragraph:
            # 1) batch 안의 모든 문단 리스트를 꺼내고
            paragraph_texts = [x['paragraph_text'] for x in batch]  # List[List[str]]
            # (옵션) paragraph_index 도 똑같이
            paragraph_idxs  = [x['paragraph_index'] for x in batch] # List[List[int]]
            
            # 2) flatten 해서 한 번에 토크나이징
            flat_paras = [p for paras in paragraph_texts for p in paras]
            tok = self.tokenizer(
                flat_paras,
                padding='longest',
                truncation=True,
                max_length=self.args.max_length,
                return_tensors='pt'
            )
            
            # 3) 다시 샘플 단위로 split
            para_counts = [len(paras) for paras in paragraph_texts]
            split_ids   = tok['input_ids'].split(para_counts,   dim=0)
            split_masks = tok['attention_mask'].split(para_counts, dim=0)
            
            # 4) 배치 내 최대 문단 수로 패딩
            max_paras = max(para_counts)
            seq_len   = tok['input_ids'].size(1)
            
            padded_ids   = []
            padded_masks = []
            for ids, masks in zip(split_ids, split_masks):
                pad_num = max_paras - ids.size(0)
                if pad_num > 0:
                    # [pad_num, seq_len] 짜리 0 패드
                    pad_ids   = torch.zeros((pad_num, seq_len), dtype=ids.dtype)
                    pad_masks = torch.zeros((pad_num, seq_len), dtype=masks.dtype)
                    ids   = torch.cat([ids,   pad_ids],   dim=0)
                    masks = torch.cat([masks, pad_masks], dim=0)
                padded_ids.append(ids)       # [num_paras_i, seq_len] → [max_paras, seq_len]
                padded_masks.append(masks)
            
            # 결과: [batch_size, max_paras, seq_len]
            item['input_ids']      = torch.stack(padded_ids,   dim=0)
            item['attention_mask'] = torch.stack(padded_masks, dim=0)
            
            # AvsH 모델을 위한 label 처리
            if labels is not None:
                # total_labels: 문서 레벨 레이블
                item['total_labels'] = labels
                # paragraph_labels: 문단 레벨 레이블 (각 문단에 문서 레이블 할당)
                paragraph_labels_list = []
                for para_count, label in zip(para_counts, labels):
                    para_labels = torch.full((para_count,), label.item(), dtype=torch.float)
                    # 패딩 부분은 -1로 설정
                    pad_num = max_paras - para_count
                    if pad_num > 0:
                        pad_labels = torch.full((pad_num,), -1.0, dtype=torch.float)
                        para_labels = torch.cat([para_labels, pad_labels])
                    paragraph_labels_list.append(para_labels)
                item['paragraph_labels'] = torch.stack(paragraph_labels_list, dim=0)
            
        else:
            # 기존 full_text 토크나이징 로직
            if self.args.add_title:
                texts = [f"{x['title']} [SEP] {x['full_text']}" for x in batch]
            else:
                texts = [x['full_text'] for x in batch]
            tok = self.tokenizer(
                texts,
                padding='longest',
                truncation=True,
                max_length=self.args.max_length,
                return_tensors='pt'
            )
            item['input_ids']      = tok['input_ids'] #[batch, seq_len]
            item['attention_mask'] = tok['attention_mask'] #[batch, seq_len]
        
        
        if self.is_first_init:
            self.is_first_init = False
            try:
                print(item['labels'])
            except:
                pass #for test
        
        return item
    