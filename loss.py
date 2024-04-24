import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, T=1):
        super(KnowledgeDistillationLoss, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        # Apply temperature scaling
        student_logits_scaled = student_logits / self.T
        teacher_logits_scaled = teacher_logits / self.T

        # Compute softmax
        student_probs = F.softmax(student_logits_scaled, dim=1)
        teacher_probs = F.softmax(teacher_logits_scaled, dim=1)

        # Compute KL divergence loss
        loss = F.kl_div(F.log_softmax(student_logits_scaled, dim=1), F.softmax(teacher_logits_scaled, dim=1), reduction='batchmean')

        return loss
    

class WaypointMSE(nn.Module):
    def __init__(self):
        super(WaypointMSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)