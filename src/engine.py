from tqdm import tqdm
import torch
import torch.nn as nn


def loss(outputs, target):
	return nn.BCEWithLogitLoss()(outputs, target)


def train(data_loader, model, optimizer, device):
	model.train()

	for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
		ids = d['ids']
		token_type_ids = d['token_type_ids']
		mask = d['mask']
		target = d['target']

		ids = ids.to(device, dtype=torch.long)
		token_type_ids = token_type_ids.to(device, dtype=torch.long)
		mask = mask.to(device, dtype=torch.long)
		target = target.to(device, dtype=torch.float)

		optimizer.zero_grad()
		outputs = model(
			ids=ids,
			mask=mask,
			token_type_ids=token_type_ids
		)

		loss = loss(outputs, target)
		loss.backward()

		optimizer.step()
		scheduler.step()


def eval(data_loader, model, device):
	model.eval()
	f_targets = []
	f_outputs = []
	with torch.no_grad():
		for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
			ids = d['ids']
			token_type_ids = d['token_type_ids']
			mask = d['mask']
			target = d['target']

			ids = ids.to(device, dtype=torch.long)
			token_type_ids = token_type_ids.to(device, dtype=torch.long)
			mask = mask.to(device, dtype=torch.long)
			target = target.to(device, dtype=torch.float)

			optimizer.zero_grad()
			outputs = model(
				ids=ids,
				mask=mask,
				token_type_ids=token_type_ids
			)

			fin_target = target.cpu().detach().numpy().to_list()
			fin_output = torch.sigmoid(outputs).cpu().detach().numpy().to_list()

			f_targets.extend(fin_target)
			f_outputs.extend(fin_output)

	return f_outputs, f_targets
