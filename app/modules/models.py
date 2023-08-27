import torch
from torch import cuda
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel

device = 'cpu'

class BertSearcher(torch.nn.Module):
    def __init__(self):
        super(BertSearcher, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.model = BertModel.from_pretrained('cointegrated/rubert-tiny2', output_attentions = False, output_hidden_states = False)

    def forward(self, text, attention_mask=None, head_mask=None, labels=None, return_dict=False):

        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=150,
            # pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt',
        )

        tokenized = encoded_dict["input_ids"].to(device)

        _, output = self.model(
            input_ids=tokenized,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=return_dict,
        )

        return output