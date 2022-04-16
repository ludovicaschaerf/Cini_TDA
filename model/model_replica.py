import torch
from torch import nn
from torchvision import models

# from scipy import sparse
        
#https://github.com/filipradenovic/cnnimageretrieval-pytorch

class ReplicaNet(torch.nn.Module):
    def __init__(self, model_name, device, pooling='avg'):
        super(ReplicaNet, self).__init__()

        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=True)
        elif model_name == "resnet152":
            model = models.resnet152(pretrained=True)
        elif model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
        elif model_name == 'resnext-101':
            model = models.resnext101_32x8d(pretrained=True)
        elif model_name == 'regnet_x_32gf':
            model = models.regnet_y_32gf(pretrained=True)
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=True)
        elif model_name == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=True)
        elif model_name == "efficientnet0":
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == "efficientnet7":
            model = models.efficientnet_b7(pretrained=True)

        
        model.fc = nn.Identity()
        
        self.non_pooled = torch.nn.Sequential(
                *(list(model.children())[:-2])
                ).to(device)
        
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1)).to(device)
        
        self.fc = model.fc.to(device)

        self.model = model.to(device)

        
    def forward(self, a, b, c):
        
        a_emb = self.model(a)
        b_emb = self.model(b)
        c_emb = self.model(c)
        
        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        b_norm = torch.div(b_emb, torch.linalg.vector_norm(b_emb))
        c_norm = torch.div(c_emb, torch.linalg.vector_norm(c_emb))
        
        return a_norm, b_norm, c_norm

    def non_pooled_forward(self, a, b, c):

        a_np = self.non_pooled(a)
        b_np = self.non_pooled(b)
        c_np = self.non_pooled(c)
        
        a_p = self.pool(a_np)
        b_p = self.pool(b_np)
        c_p = self.pool(c_np)
        
        a_emb = self.fc(a_p)
        b_emb = self.fc(b_p)
        c_emb = self.fc(c_p)

        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        b_norm = torch.div(b_emb, torch.linalg.vector_norm(b_emb))
        c_norm = torch.div(c_emb, torch.linalg.vector_norm(c_emb))
        

        return a_np, b_np, c_np, a_norm, b_norm, c_norm

    def size(self, a):
        size = a.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, a):
        a_emb = self.model(a)
        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        return a_norm
