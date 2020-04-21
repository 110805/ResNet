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
train_data = RetinopathyLoader(root='/home/ubuntu/Retinopathy_detection/data/', mode='train')
test_data = RetinopathyLoader(root='/home/ubuntu/Retinopathy_detection/data/', mode='test')
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
ResNet18_1 = torchvision.models.resnet18(pretrained=False)
ResNet18_1.fc = nn.Linear(512, 5)
ResNet18_2 = torchvision.models.resnet18(pretrained=True)
ResNet18_2.fc = nn.Linear(512, 5)

ResNet18_1.load_state_dict(torch.load('0.pkl'))
ResNet18_2.load_state_dict(torch.load('1.pkl'))

device = torch.device('cuda')
p = []
l = []
ResNet18_1.to(device)
for idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = ResNet18_1(inputs.float())

    _, preds = torch.max(outputs, 1) # the second return of max is the return of argmax
    p += preds.tolist()
    l += labels.data.flatten().tolist()

array = confusion_matrix(l,p,normalize='all')
df_cm = pd.DataFrame(array, range(5), range(5))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.savefig('con.png')
