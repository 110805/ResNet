import torch
import torchvision.models
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 4

test_data = RetinopathyLoader(root='/home/ubuntu/Retinopathy_detection/data/', mode='test')
test_loader = DataLoader(test_data, batch_size=batch_size)
ResNet18_1 = torchvision.models.resnet18(pretrained=False)
ResNet18_1.fc = nn.Linear(512, 5)
ResNet18_2 = torchvision.models.resnet18(pretrained=True)
ResNet18_2.fc = nn.Linear(512, 5)
ResNet50_1 = torchvision.models.resnet50(pretrained=False)
ResNet50_1.fc = nn.Linear(2048, 5)
ResNet50_2 = torchvision.models.resnet50(pretrained=True)
ResNet50_2.fc = nn.Linear(2048, 5) 

ResNet18_1.load_state_dict(torch.load('0.pkl'))
ResNet18_2.load_state_dict(torch.load('1.pkl'))
#ResNet50_1.load_state_dict(torch.load('2.pkl'))
#ResNet50_2.load_state_dict(torch.load('3.pkl'))

device = torch.device('cuda')
def restore(model, i):
    p = []
    l = []
    correct = 0
    model.to(device)
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs.float())

        _, preds = torch.max(outputs, 1) # the second return of max is the return of argmax
        p += preds.tolist()
        l += labels.data.flatten().tolist()
        correct += torch.sum(preds == labels.data.flatten())
    
    acc = 100*correct/7025
    print('Acc = {}%'.format(acc))
    mat = confusion_matrix(l,p,normalize='all')
    df_cm = pd.DataFrame(mat, range(5), range(5))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.savefig('{}.png'.format(i))

restore(ResNet18_1, 1)
restore(ResNet18_2, 2)
#restore(ResNet_50_1, 3)
#restore(ResNet_50_1, 4)
