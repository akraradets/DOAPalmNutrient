import torchvision.models as models
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy,copy

# the magic number
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

preprocess_augment = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

preprocess = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

from myDataset import PalmNutriDataset
ground_truth = 'dataset/gt.csv'
full_train_dataset = PalmNutriDataset(ground_truth=ground_truth, img_dir='dataset', sample_set='all_17', target='n')
print(len(full_train_dataset)) # => 900

train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [1600,318])
train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = preprocess_augment
val_dataset.dataset.transform = preprocess

BATCH_SIZE=16
NUM_WORKERS=2
train_dataloader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE,shuffle=True , num_workers=NUM_WORKERS)
val_dataloader   = torch.utils.data.DataLoader(val_dataset  , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)
# test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)

from trainer import trainer
dataloaders = {'train': train_dataloader,'val':val_dataloader}
# Set device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.alexnet()
model.classifier[6] = torch.nn.Linear(in_features=4096,out_features=1)
# Optimizer and loss function
criterion = nn.MSELoss()
params_to_update = model.parameters()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(params_to_update, lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
t = trainer(device,criterion, optimizer,scheduler)
model = t.train(model, dataloaders, num_epochs=150, weights_name='alex_sgd_0.01')
# t.test(model,test_dataloader)