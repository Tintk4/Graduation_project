# import os
# import sys
# import collections
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
# from ..models.classifier import Classifier
#
# # bert base model from https://huggingface.co/bert-base-uncased
# # bert large model from https://huggingface.co/bert-large-uncased
#
#
# class BertTextClf(nn.Module):
#     def __init__(self, args):
#         super(BertTextClf, self).__init__()
#         self.text_encoder = BertModel.from_pretrained(args.bert_model)
#         self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)
#
#     def forward(self, txt, mask, segment):
#         x = self.text_encoder(txt, token_type_ids=segment,attention_mask=mask,).last_hidden_state[:, 0, :]
#         return self.clf(x)
#
#     # def forward(self, text):
#     #     """
#     #     text: (batch_size, 3, seq_len)
#     #     3: input_ids, input_mask, segment_ids
#     #     input_ids: input_ids,
#     #     input_mask: attention_mask,
#     #     segment_ids: token_type_ids
#     #     """
#     #     input_ids = torch.squeeze(text[0], 1)
#     #     input_mask = torch.squeeze(text[2], 1)
#     #     segment_ids = torch.squeeze(text[1], 1)
#     #     # input_ids, input_mask, segment_ids = input_ids, attention_mask, token_type_ids
#     #     last_hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
#     #
#     #     return last_hidden_states
#
#
# # if __name__ == "__main__":
# #     text_normal = TextEncoder()

import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from ..models.classifier import Classifier


class BertTextClf(nn.Module):
    def __init__(self, args):
        super(BertTextClf, self).__init__()

        # 方案1：直接指定本地路径（最简单）
        local_model_path = "D:/models/bert-base-uncased"  # 替换为您的实际路径

        # 方案2：通过args传递本地路径（更灵活）
        if hasattr(args, 'local_bert_path') and args.local_bert_path:
            local_model_path = args.local_bert_path

        # 尝试加载本地模型，失败则回退到在线加载
        try:
            print(f"尝试加载本地BERT模型: {local_model_path}")
            self.text_encoder = BertModel.from_pretrained(local_model_path)
            print("✓ 本地模型加载成功")
        except Exception as e:
            print(f"✗ 本地模型加载失败: {e}")
            print(f"回退到在线加载: {args.bert_model}")
            self.text_encoder = BertModel.from_pretrained(args.bert_model)

        self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)

    def forward(self, txt, mask, segment):
        x = self.text_encoder(txt, token_type_ids=segment, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.clf(x)