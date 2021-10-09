from torch.optim import optimizer
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
full_train_dataset = PalmNutriDataset(ground_truth=ground_truth, img_dir='dataset', sample_set='k17', target='k')
print(len(full_train_dataset))
# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [750,150])
train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = preprocess_augment
val_dataset.dataset.transform = preprocess

BATCH_SIZE=128
NUM_WORKERS=4
train_dataloader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE,shuffle=True , num_workers=NUM_WORKERS)
val_dataloader   = torch.utils.data.DataLoader(val_dataset  , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)
# test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)

from trainer import trainer
dataloaders = {'train': train_dataloader,'val':val_dataloader}
# Set device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.mobilenet_v3_large()
model.classifier[3] = torch.nn.Linear(in_features=1280,out_features=1,bias=True)
# model = models.mobilenet_v3_small()
# model.classifier[3] = torch.nn.Linear(in_features=1024,out_features=1,bias=True)
# Optimizer and loss function
criterion = nn.MSELoss()
params_to_update = model.parameters()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(params_to_update, lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
t = trainer(device,criterion, optimizer,scheduler)
model = t.train(model, dataloaders, num_epochs=150, weights_name='mobilenet_k17')
# t.test(model,test_dataloader)