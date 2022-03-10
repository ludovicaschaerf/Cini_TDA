import torch
from torch import nn
from torchvision import models
from scipy import sparse
        

class ReplicaNet(torch.nn.Module):
    def __init__(self, device):
        super(ReplicaNet, self).__init__()

        efficientnet_b7 = models.efficientnet_b0(pretrained=True, progress=False)
        efficientnet_b7.classifier = nn.Sequential()
        
        self.efficientnet = efficientnet_b7.to(device)
        
    def forward(self, a, b, c):
        # if self.training:
        #    print('training')
        # else:
        #    print('eval')

        a_emb = self.efficientnet(a)
        b_emb = self.efficientnet(b)
        c_emb = self.efficientnet(c)
        
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
        a_emb = self.efficientnet(a)
        a_norm = torch.div(a_emb, torch.linalg.vector_norm(a_emb))
        return sparse.csr_matrix(a_norm.cpu().detach().numpy())

    def evaluate(self, set_b, set_c):
        intersection = set_b.intersection(set_c)
        return len(list(intersection)) / min(len(list(set_c)), len(list(set_b)))

