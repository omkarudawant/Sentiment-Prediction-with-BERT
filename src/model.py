import transformers
import torch.nn as nn
import config


class BertBaseUncased(nn.Module):
	def __init__(self):
		super(BertBaseUncased, self).__init__()
		self.bert = transformers.BertModel.from_pretrained(
			config.BERT_PATH)
		self.bert_drop = nn.Dropout(0.2)
		self.out = nn.Linear(768, 1)

	def forward(self, id, mask, token_type_id):
		hidden_states, pooled_out = self.bert(
			id,
			attention_mask=mask,
			token_type_ids=token_type_id
			)
		bert_out = self.bert_drop(pooled_out)
		output = self.out(bert_out)
		return output
 