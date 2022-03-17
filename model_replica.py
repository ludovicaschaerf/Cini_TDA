import torch
from torch import nn
from torchvision import models
from scipy import sparse
        

class ReplicaNet(torch.nn.Module):
    def __init__(self, device):
        super(ReplicaNet, self).__init__()

        model = models.resnet50(pretrained=True)
        newmodel = torch.nn.Sequential(
            *(list(model.children())[:-2]), nn.AdaptiveMaxPool2d((1,1))
        )

        self.model = newmodel.to(device)
        
    def forward(self, a, b, c):
        
        a_emb = self.model(a)
        b_emb = self.model(b)
        c_emb = self.model(c)
        
        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        b_norm = torch.div(b_emb, torch.linalg.vector_norm(b_emb))
        c_norm = torch.div(c_emb, torch.linalg.vector_norm(c_emb))

        return a_norm, b_norm, c_norm

    def size(self, a):
        size = a.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, a):
        a_emb = self.model(a)
        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        return sparse.csr_matrix(a_norm.cpu().detach().numpy())

    def evaluate(self, set_b, set_c):
        intersection = set_b.intersection(set_c)
        return len(list(intersection)) / min(len(list(set_c)), len(list(set_b)))

