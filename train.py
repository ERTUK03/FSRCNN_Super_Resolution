import torch
import get_dataset, prepare_dataset, model_builder, engine, utils

config_data = utils.load_config()

URL = config_data['url']
FILENAME = config_data['filename']
DIR_NAME = config_data['dir_name']
BATCH_SIZE = config_data['batch_size']
D = config_data['d']
S = config_data['s']
M = config_data['m']
LEARNING_RATE = config_data['l_r']
EPOCHS = config_data['epochs']
MODEL_NAME = config_data['name']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

get_dataset.download_dataset(URL, FILENAME, DIR_NAME)

train_dataloader, test_dataloader = prepare_dataset.get_dataloaders(DIR_NAME, BATCH_SIZE)

model = model_builder.FSRCNN(d=D, s=S, m=M).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params':model.feature_extraction.parameters()},
    {'params':model.shrinking.parameters()},
    {'params':model.non_linear_mapping.parameters()},
    {'params':model.expanding.parameters()},
    {'params':model.deconvolution.parameters(), 'lr':LEARNING_RATE/10}
    ], lr=LEARNING_RATE)

engine.train(EPOCHS, model, optimizer, criterion, train_dataloader, test_dataloader, MODEL_NAME, device)
