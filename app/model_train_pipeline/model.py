import numpy as np
import pandas as pd

import torch
from torch import cuda
from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel


class BertScorer(torch.nn.Module):
    def __init__(self):
        super(BertScorer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.model = BertModel.from_pretrained('cointegrated/rubert-tiny2', output_attentions=False,
                                               output_hidden_states=False)
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.fc1 = nn.Linear(624, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, text, target, attention_mask=None, head_mask=None, labels=None, return_dict=False):
        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=150,
            # pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt',
        )["input_ids"].to(self.device)

        tokenized_target = self.tokenizer.encode_plus(
            target,
            add_special_tokens=True,
            truncation=True,
            max_length=150,
            # pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt',
        )["input_ids"].to(self.device)

        _, output_text = self.model(
            input_ids=tokenized_text,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=return_dict,
        )

        _, output_target = self.model(
            input_ids=tokenized_target,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=return_dict,
        )

        emb = torch.cat([output_text, output_target], dim=1)
        emb = F.relu(self.fc1(emb))
        emb = F.relu(self.fc2(emb))
        emb = F.relu(self.fc3(emb))
        emb = F.relu(self.fc4(emb))
        output = F.sigmoid(self.fc5(emb))

        return output
