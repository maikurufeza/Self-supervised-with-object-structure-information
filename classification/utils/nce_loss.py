import torch
import torch.nn.functional as F


class Info_NCE_Loss:
    """ function of NCE loss

        Attributes:
            num_positive->int: number of positive images
            device: the device index
            temperature: the temperature parameter (\tau)
        """
    def __init__(self, num_positive, temperature, device):
        self.num_positive = num_positive
        self.device = device
        self.temperature = temperature
    
    def __call__(self, features):
        N = int(features.shape[0]/self.num_positive)
        labels = torch.cat([torch.arange(N) for i in range(self.num_positive)], dim=0) # [0,...N-1,0,...N-1,...(num_positive times)]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # (N*num_positive, N*num_positive)
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        return nce(logits,self.num_positive)
    """labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, labels"""


def nce(inp,t):
    positive = torch.sum(torch.exp(inp[:,0:t-1]),dim=1)
    total = torch.sum(torch.exp(inp),dim=1)
    return -torch.mean(torch.log(positive/total))
