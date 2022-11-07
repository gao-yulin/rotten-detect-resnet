import torch
import torch.nn as nn
class binaryResnet50(nn.Module):

    def __init__(self):
        super(binaryResnet50, self).__init__()

        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.conv = nn.Sequential(*list(resnet50.children())[0:9])
        self.fc = nn.Linear(in_features=2048, out_features=2, bias=True)


    def forward(self, x):
        out1 = self.conv(x)
        out1 = out1.squeeze()
        out2 = self.fc(out1)
        return out2
