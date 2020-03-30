# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:28:02 2020

@author: Kitt Miller
"""

import torch
import numpy as np
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
import tarfile
import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer

## utils
def imshow(image_tensor, mean, std, title=None):
    """
    Imshow for normalized Tensors.
    Useful to visualize data from data loader
    """

    image = image_tensor.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def accuracy(output, target):

    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    accuracy = correct/total

    return accuracy


class Tracker:

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
                
## Plotting                
def plotter():
    path_adapt = "log_adaptation.pth"

    # Load logs
    log_adapt = torch.load(path_adapt)

    # compute mean for each epoch
    adapt = {
        'classification_loss': torch.FloatTensor(log_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    # Add the first 0 value
    adapt['target_accuracy'] = np.insert(adapt['target_accuracy'], 0, 0)
    adapt['source_accuracy'] = np.insert(adapt['source_accuracy'], 0, 0)


    axes = plt.gca()
    axes.set_ylim([0, 1.1])

    l1, = plt.plot(adapt['target_accuracy'], label="test acc. w/ coral loss", marker='*')
    l3, = plt.plot(adapt['source_accuracy'], label="training acc. w/ coral loss", marker='^')

    plt.legend(handles=[l1, l3], loc=4)
    fig_acc = plt.gcf()
    plt.show()
    plt.figure()
    fig_acc.savefig('accuracies.pdf', dpi=1000)

    # Classification loss and CORAL loss for training w/ CORAL loss

    axes = plt.gca()
    axes.set_ylim([0, 0.5])

    l5, = plt.plot(adapt['classification_loss'], label="classification loss", marker='.')
    l6, = plt.plot(adapt['coral_loss'], label="coral loss", marker='*')

    plt.legend(handles=[l5, l6], loc=1)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()
    fig_acc.savefig('losses_adapt.pdf', dpi=1000)

    
## Model
class Net(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(Net, self).__init__()

        # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        self.model = models.densenet161(pretrained=pretrained, num_classes=num_classes)

        # if we want to feed 448x448 images
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        # In case we want to apply the loss to any other layer than the last
        # we need a forward hook on that layer
        # def save_features_layer_x(module, input, output):
        #     self.layer_x = output

        # This is a forward hook. Is executed each time forward is executed
        # self.model.layer4.register_forward_hook(save_features_layer_x)

    def forward(self, x):
        out = self.model(x)
        return out  # , self.layer_x
    
## Dataloading
def get_loader(name_dataset, batch_size, mean_std, train=True):

    # Computed with compute_mean_std.py
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_std[name_dataset]['mean'],
                                  std=mean_std[name_dataset]['std'])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_std[name_dataset]['mean'],
                                  std=mean_std[name_dataset]['std'])])
        }
    

    dataset = datasets.ImageFolder(root='./data/%s' % name_dataset,
                                   transform=transform['train' if train else 'test'])
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=4)
    return dataset_loader

## Coral
def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

## Mean + Stdev
def compute_mean_std(path_dataset):
    """
    Compute mean and standard deviation of an image dataset.
    Acknowledgment : http://forums.fast.ai/t/image-normalization-in-pytorch/7534
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=path_dataset,
                                   transform=transform)
    # Choose a large batch size to better approximate. Optimally load the dataset entirely on memory.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)

    pop_mean = []
    pop_std = []

    for i, data in enumerate(data_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,) -> 3 channels
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)

    values = {
        'mean': pop_mean,
        'std': pop_std
    }

    return values


def dataset_stats():
    mean_std = {}
    for dataset in ['labeled', 'unlabeled', 'test']:
        # Construct path
        dataset_path = './data/%s' % dataset
        values = compute_mean_std(dataset_path)
        # Add values to dict
        mean_std[dataset] = values

    return mean_std


    
## Training
def train(model, optimizer, source_loader, target_loader, tracker, args, epoch=0):

    model.train()
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

    # Trackers to monitor classification and CORAL loss
    classification_loss_tracker = tracker.track('classification_loss', tracker_class(**tracker_params))
    coral_loss_tracker = tracker.track('CORAL_loss', tracker_class(**tracker_params))

    min_n_batches = min(len(source_loader), len(target_loader))

    tq = tqdm(range(min_n_batches), desc='{} E{:03d}'.format('Training + Adaptation', epoch), ncols=0)

    for _ in tq:

        source_data, source_label = next(iter(source_loader))
        target_data, _ = next(iter(target_loader))  # Unsupervised Domain Adaptation

        source_data, source_label = Variable(source_data.to(device=args.device)), Variable(source_label.to(device=args.device))
        target_data = Variable(target_data.to(device=args.device))

        optimizer.zero_grad()

        out_source = model(source_data)
        out_target = model(target_data)

        classification_loss = F.cross_entropy(out_source, source_label)

        # This is where the magic happens
        coral_loss = coral(out_source, out_target)
        composite_loss = classification_loss + args.lambda_coral * coral_loss

        composite_loss.backward()
        optimizer.step()

        classification_loss_tracker.append(classification_loss.item())
        coral_loss_tracker.append(coral_loss.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(classification_loss=fmt(classification_loss_tracker.mean.value),
                       coral_loss=fmt(coral_loss_tracker.mean.value))


def evaluate(model, data_loader, dataset_name, tracker, args, epoch=0):
    model.eval()

    tracker_class, tracker_params = tracker.MeanMonitor, {}
    acc_tracker = tracker.track('{}_accuracy'.format(dataset_name), tracker_class(**tracker_params))

    loader = tqdm(data_loader, desc='{} E{:03d}'.format('Evaluating on %s' % dataset_name, epoch), ncols=0)

    accuracies = []
    with torch.no_grad():
        for target_data, target_label in loader:
            target_data = Variable(target_data.to(device=args.device))
            target_label = Variable(target_label.to(device=args.device))

            output = model(target_data)

            accuracies.append(accuracy(output, target_label))

            acc_tracker.append(sum(accuracies)/len(accuracies))
            fmt = '{:.4f}'.format
            loader.set_postfix(accuracy=fmt(acc_tracker.mean.value))


def main():

    # Paper: In the training phase, we set the batch size to 128,
    # base learning rate to 10−3, weight decay to 5×10−4, and momentum to 0.9

    parser = argparse.ArgumentParser(description='Train - Evaluate DeepCORAL model')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning Rate')
    parser.add_argument('--decay', default=5e-4,
                        help='Decay of the learning rate')
    parser.add_argument('--momentum', default=0.9,
                        help="Optimizer's momentum")
    parser.add_argument('--lambda_coral', type=float, default=0.8,
                        help="Weight that trades off the adaptation with "
                             "classification accuracy on the source domain")
    parser.add_argument('--source', default='labeled',
                        help="Source Domain (dataset)")
    parser.add_argument('--target', default='unlabeled',
                        help="Target Domain (dataset)")
    parser.add_argument('--test', default='test',
                        help="Test/Validation Data (dataset)")
    parser.add_argument('--naming', default='unnamed',
                        help="Model save name")
    parser.add_argument('--bucket', default='domaintestkitt',
                        help="S3 bucket for output file")
    
    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        print('CUDA enabled')
    else:
        args.device = torch.device('cpu')
        print('CUDA disabled or not available')
    
    stats = dataset_stats()
    source_train_loader = get_loader(name_dataset=args.source, batch_size=args.batch_size, mean_std =stats, train=True)
    target_train_loader = get_loader(name_dataset=args.target, batch_size=args.batch_size, mean_std =stats, train=True)

    source_evaluate_loader = get_loader(name_dataset=args.source, batch_size=args.batch_size, mean_std =stats, train=False)
    target_evaluate_loader = get_loader(name_dataset=args.test, batch_size=args.batch_size, mean_std =stats, train=False)

    n_classes = len(source_train_loader.dataset.classes)

    # ~ Paper : "We initialized the other layers with the parameters pre-trained on ImageNet"
    model = models.densenet169(pretrained=True)
    # ~ Paper : The dimension of last fully connected layer (fc8) was set to the number of categories (31)
    model.classifier = nn.Linear(1664, n_classes)
    # ~ Paper : and initialized with N(0, 0.005)
    torch.nn.init.normal_(model.classifier.weight, mean=0, std=5e-3)

    # Initialize bias to small constant number (http://cs231n.github.io/neural-networks-2/#init)
    model.classifier.bias.data.fill_(0.01)

    model = model.to(device=args.device)

    # ~ Paper : "The learning rate of fc8 is set to 10 times the other layers as it was training from scratch."
    optimizer = torch.optim.SGD([
        {'params':  model.features.parameters()},
        {'params': model.classifier.parameters(), 'lr': 5 * args.lr}
    ], lr=args.lr, momentum=args.momentum)  # if not specified, the default lr is used

    tracker = Tracker()

    for i in range(args.epochs):
        train(model, optimizer, source_train_loader, target_train_loader, tracker, args, i)
        evaluate(model, source_evaluate_loader, 'source', tracker, args, i)
        evaluate(model, target_evaluate_loader, 'target', tracker, args, i)

    # Save logged classification loss, coral loss, source accuracy, target accuracy
    torch.save(tracker.to_dict(), "log_adaptation.pth")
    torch.save(model, str(str(args.naming) +'_model.pt'))
    with open('model.pth', 'wb') as f:
        torch.save(model.state_dict(), f)
    
    for i in stats.keys():
        for j in ['mean', 'std']:
            stats[i][j] = np.ndarray.tolist(stats[i][j])    
    model_data = {
            'num_classes': n_classes,
            'classes':source_train_loader.dataset.classes,
            'data_stats': stats,
            }
    with open('model_data.json', 'w') as fp:
        json.dump(model_data, fp)
    
    tar = tarfile.open("output.tar.gz", "w:gz")
    for name in ["model_data.json", "log_adaptation.pth", str(str(args.naming) +'_model.pt'), 'model.pth']:
        tar.add(name)
    tar.close()
    
    role = *<REPLACE WITH YOUR SAGEMAKER ENABLE ROLE ARN>*
    s3client = boto3.client('s3')
    with open('output.tar.gz', "rb") as f:
        s3client.upload_fileobj(f, args.bucket, 'output.tar.gz')
    
    class JSONPredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)
    
    model = PyTorchModel(model_data="s3://{}/output.tar.gz".format(args.bucket),
                     role=role,
                     framework_version='1.4.0',
                     entry_point='serve_demo.py',
                     predictor_cls=JSONPredictor)
    
    predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
    
    with open("endpoint.txt", "w") as write_file:
        write_file.write(predictor.endpoint)
    
if __name__ == '__main__':
    main()
    plotter()
