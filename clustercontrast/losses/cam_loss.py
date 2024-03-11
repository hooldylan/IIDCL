import torch
import torch.nn as nn
import torch.nn.functional as F
class InterCamProxy(nn.Module):
    """ Camera-aware proxy with inter-camera contrastive learning """
    def __init__(self, num_features, num_samples, num_hards=50, temp=0.07):
        super(InterCamProxy, self).__init__()
        # 特征数量
        self.num_features = num_features  # D
        # 样本数量
        self.num_samples = num_samples  # N
        # hard数量
        self.num_hards = num_hards
        # 在softmax层外加了log函数
        self.logsoftmax = nn.LogSoftmax(dim=0)
        # 温度超参数
        self.temp = temp
        # 该方法的作用是定义一组参数模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer('proxy', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.zeros(num_samples).long())
        self.register_buffer('cids', torch.zeros(num_samples).long())

    """ Inter-camera contrastive loss """   
    
    def forward(self, inputs, targets, cams):
        B, D = inputs.shape
        # F.normalize 每个数都除以二范数
        inputs = F.normalize(inputs, dim=1).cuda()  # B * D
        sims = inputs @ self.proxy.T  # B * N
        sims /= self.temp
        temp_sims = sims.detach().clone()

        loss = torch.tensor(0.).cuda()
        for i in range(B):
            pos_mask = (targets[i] == self.pids).float() * (cams[i] != self.cids).float()
            neg_mask = (targets[i] != self.pids).float()
            pos_idx = torch.nonzero(pos_mask > 0).squeeze(-1)
            if len(pos_idx) == 0:
                continue
            hard_neg_idx = torch.sort(temp_sims[i] + (-9999999.) * (1.-neg_mask), descending=True).indices[:self.num_hards]
            sims_i = sims[i, torch.cat([pos_idx, hard_neg_idx])]
            targets_i = torch.zeros(len(sims_i)).cuda()
            targets_i[:len(pos_idx)] = 1.0 / len(pos_idx)
            loss += - (targets_i * self.logsoftmax(sims_i)).sum()

        loss /= B
        return loss