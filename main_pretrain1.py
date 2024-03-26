import argparse
import os
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import wandb

# Remove multi-GPU related imports

# Remove distributed training related imports

# Remove GatherLayer and gather_from_all functions

# Remove lamda_scheduler and cosine_scheduler functions

# Remove main_worker function

# Remove all references to args.gpu, args.rank, and args.distributed

# Remove all distributed training logic

# Remove lamda_inv and momentum_schedule related code

# Remove learning rate scheduling related code

# Remove model and optimizer setup logic for multi-GPU

# Remove distributed data loading logic

# Remove save_checkpoint function

# Remove train_sampler and related logic

# Remove all references to distributed training and synchronization

# Define main function
def main():
    # Define and parse arguments
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.project_name, entity="alicelouis", config=args.__dict__, name=args.run_name)
        
    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Model setup
    model = models.__dict__[args.arch](pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.cuda()  # Move model to GPU
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data loading and augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(args.data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

# Define train function
def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Log training stats
        if batch_idx % args.print_freq == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

# Define argument parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='ImageNet', help='dataset')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('--num-classes', default=1000, type=int, help='number of classes in the dataset')
parser.add_argument('--batch-size', default=1024, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--wandb', action='store_true', help="whether to use wandb")
parser.add_argument('--project-name', default='project_name', type=str, help='wandb project name')
parser.add_argument('--run-name', default='run_name', type=str, help='wandb run name')

if __name__ == '__main__':
    main()
