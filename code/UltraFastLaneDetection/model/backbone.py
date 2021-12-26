import torch,pdb
import torchvision
import torch.nn.modules
from functions import get_RedNetInvolution


class RedNetInvolution(torch.nn.Module):
    def __init__(self, layers, pretrained, finetune, checkpoint_name, frozen_blocks):
        super(RedNetInvolution,self).__init__()
        
        model = get_RedNetInvolution(depth=layers,
                                     pretrained=pretrained,
                                     finetune=finetune,
                                     checkpoint_name=checkpoint_name,
                                     frozen_blocks=frozen_blocks)
        
        self.stem = model.stem
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4
