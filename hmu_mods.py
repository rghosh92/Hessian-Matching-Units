# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:37:40 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPNet_probing(nn.Module):
    def __init__(self, input_channels, block, num_blocks, num_classes=10, madden = 128):
        super().__init__()
        self.in_planes = madden

        self.conv1 = nn.Conv2d(input_channels, madden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(madden)

        self.layer1 = self._make_layer(block, madden, num_blocks[0])
        self.layer2 = self._make_layer(block, madden, num_blocks[1])

        # Probes (128 → num_classes)
        self.probe1 = nn.Linear(madden, num_classes)
        self.probe2 = nn.Linear(madden, num_classes)
        self.probe3 = nn.Linear(madden, num_classes)

        self.linear = nn.Linear(madden, num_classes)
        self.drop = nn.Dropout(0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(-1).unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        feat1 = out.view(out.size(0), -1)   # already flat

        out = self.layer1(out)
        feat2 = out.view(out.size(0), -1)

        out = self.layer2(out)
        feat3 = out.view(out.size(0), -1)

        out = self.drop(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, (feat1, feat2, feat3)


class ResMLPNet_synthetic(nn.Module):
    def __init__(self, input_channels, block, num_blocks, num_classes=10, madden = 128):
        super().__init__()
        self.in_planes = madden

        self.conv1 = nn.Conv2d(input_channels, madden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(madden)

        self.layer1 = self._make_layer(block, madden, num_blocks[0])
        self.layer2 = self._make_layer(block, madden, num_blocks[1])

        self.linear = nn.Linear(madden, num_classes)
        self.drop = nn.Dropout(0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        feat1 = out.view(out.size(0), -1)   # already flat

        out = self.layer1(out)
        feat2 = out.view(out.size(0), -1)

        out = self.layer2(out)
        feat3 = out.view(out.size(0), -1)

        out = self.drop(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, 0
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BasicBlock_HMU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,k=3):
        super(BasicBlock_HMU, self).__init__()
        
        self.hmu1 = HMULayer(in_planes, planes, k)
        self.bn1 = nn.BatchNorm1d(planes)
        self.hmu2 = HMULayer(planes, planes, k)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                HMULayer(in_planes, self.expansion*planes, k),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = x.squeeze()
        out = self.bn1(self.hmu1(x))
        out = self.bn2(self.hmu2(out))
        
        out += self.shortcut(x)
        # out = F.relu(out)
        return out

class HMU(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d = d
        self.k = k
        
        # Isotropic baseline curvature (lambda)
        self.lambda_base = nn.Parameter(torch.ones(1))
        
        # k directions (v_i). Parameterized as (k x d)
        self.v = nn.Parameter(torch.randn(k, d) * 0.1)
        
        # Sign toggles (eta_i). Fixed or learnable? 
        # For this test, we allow them to be learnable weights to determine sign.

    def get_hessian(self):
        """
        Computes the Hessian matrix W = lambda*I + sum(omega_i * v_i * v_i^T)
        """
        # Start with the isotropic baseline
        W = self.lambda_base * torch.eye(self.d)
        
        # Add the k low-rank updates
        # v: (k, d), omega: (k) -> sum of outer products
        # Using einsum for a clean 'sum of omega_i * outer(v_i, v_i)'
        low_rank_term = torch.einsum('ki,kj->ij', self.v, self.v)
        
        return W + low_rank_term

    def forward(self, x, mu):
        diff = x - mu  # (batch, d)
        W = self.get_hessian()  # (d, d)
        # print(W.shape)
        # Mathematical expression: x_i * W_ij * x_j (summed over i, j per batch)
        # 'bi, ij, bj -> b' 
        # b = batch, i = d (left), j = d (right)
        quad_form = torch.einsum('bi,ij,bj->b', diff, W, diff)
        
        return torch.exp(-quad_form / self.d)   
    




class HMULayer_omega(nn.Module):
    def __init__(self, in_features, out_features, k,Type = 'exp', lambda_mult = 0.01):
        super().__init__()
        self.d = in_features     # Input dimensionality
        self.n = out_features    # Number of units (neurons)
        self.k = k              # Rank of curvature update
        self.type = Type
        self.lambda_mult = lambda_mult
        # 1. Centers for each unit: (N x d)
        self.mu = nn.Parameter(torch.zeros(self.n, self.d))
        
        # 2. Isotropic baselines for each unit: (N)
        self.lambda_base = nn.Parameter(torch.ones(self.n))
        
        # 3. k directions per unit: (N x k x d)
        if self.k>0:
            self.v = nn.Parameter(torch.zeros(self.n, self.k, self.d))
        
        self.omega = nn.Parameter(torch.zeros(self.n, self.k))

        self.init_params()
    
    def init_params(self):
        
        # 1. Centers: Spread them out to cover the input space
        # Using a wider spread prevents units from clumping at the origin
        self.mu = nn.Parameter(torch.Tensor(self.n, self.d))
        nn.init.uniform_(self.mu, 0.0, 1.0) 
        
        # 2. Isotropic baseline: Start at 1.0 
        # This keeps the initial "effective dimension" manageable
        self.lambda_base = nn.Parameter(torch.rand(self.n)*self.lambda_mult)
        # nn.Parameter(torch.randn(self.n, self.k) * 0.01)
        nn.init.uniform_(self.omega, 0.1,0.5)
        # 3. k-vectors: Use orthogonal initialization per unit
        if self.k>0:
            for i in range(self.n):
                # Orthogonal init ensures the k vectors don't overlap initially
                # We scale by 0.1 so the low-rank part doesn't dominate immediately
                nn.init.orthogonal_(self.v.data[i], gain=1)
    
        # Optional: Small random noise to lambda to break unit symmetry
        # with torch.no_grad():
        #     self.lambda_base.add_(torch.randn(self.n) * 0.01)
            

    def get_hessians(self):
        """
        Computes the Hessian matrix W for each of the N units.
        Returns: (N x d x d)
        """
        # Create identity matrices for all N units: (N x d x d)
        I = torch.eye(self.d, device=self.mu.device).unsqueeze(0).expand(self.n, -1, -1)
        
        # Baseline part: multiply each identity by its unit's lambda
        W = self.lambda_base.view(self.n, 1, 1) * I
        
        # Low-rank part: Sum omega_nk * outer(v_nk, v_nk)
        # n: unit index, k: rank index, i/j: dim indices
        if self.k>0:
            low_rank_term = torch.einsum('nk,nki,nkj->nij', self.omega, self.v, self.v)
            return  W + low_rank_term
        else:
            return W

    def forward(self, x):
        """
        x: (batch, d)
        returns: (batch, n)
        """
        # 1. Compute displacements for all pairs: (batch, n, d)
        # x is (B, 1, d), mu is (1, N, d) -> diff is (B, N, d)
        x = x.squeeze()
        eps = 1e-4

        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
        # sample_std = diff.norm(dim=2, keepdim=True) + eps
        
        # # Displacement normalized by each sample's std
        # diff = diff / sample_std
        
        # 2. Get all Hessians: (N, d, d)
        W = self.get_hessians()
        
        # 3. Compute the quadratic form: (batch-i, unit-n)^T * W_n * (batch-i, unit-n)
        # b: batch, n: unit, i: d, j: d
        # We want a result of shape (b, n)
        quad_form = torch.einsum('bni,nij,bnj->bn', diff, W, diff)
        
        # 4. Activation
        if self.type == 'no_exp':
            return -quad_form 
        # print(self.lambda_base.mean())
        return torch.exp(-quad_form / self.d)

    def extra_repr(self):
        return f'in_features={self.d}, out_features={self.n}, rank_k={self.k}'
    
class HMULayer(nn.Module):
    def __init__(self, in_features, out_features, k,Type = 'exp', lambda_mult = 0.01):
        super().__init__()
        self.d = in_features     # Input dimensionality
        self.n = out_features    # Number of units (neurons)
        self.k = k              # Rank of curvature update
        self.type = Type
        self.lambda_mult = lambda_mult
        # 1. Centers for each unit: (N x d)
        self.mu = nn.Parameter(torch.zeros(self.n, self.d))
        
        # 2. Isotropic baselines for each unit: (N)
        self.lambda_base = nn.Parameter(torch.ones(self.n))
        # 3. k directions per unit: (N x k x d)
        if self.k>0:
            self.v = nn.Parameter(torch.zeros(self.n, self.k, self.d))
        
        
        self.init_params()
    
    def init_params(self):
        
        # 1. Centers: Spread them out to cover the input space
        # Using a wider spread prevents units from clumping at the origin
        nn.init.uniform_(self.mu, 0.0, 1.0) 
        
        # 2. Isotropic baseline: Start at 1.0 
        # This keeps the initial "effective dimension" manageable
        if self.k>0:
            self.lambda_base = nn.Parameter(torch.rand(self.n)*0.01)
        else:
            self.lambda_base = nn.Parameter(torch.rand(self.n))
        
        # 3. k-vectors: Use orthogonal initialization per unit
        if self.k>0:
            for i in range(self.n):
                # Orthogonal init ensures the k vectors don't overlap initially
                # We scale by 0.1 so the low-rank part doesn't dominate immediately
                nn.init.orthogonal_(self.v.data[i], gain=1)
    
        # Optional: Small random noise to lambda to break unit symmetry
        # with torch.no_grad():
        #     self.lambda_base.add_(torch.randn(self.n) * 0.01)
            

    def get_hessians(self):
        """
        Computes the Hessian matrix W for each of the N units.
        Returns: (N x d x d)
        """
        # Create identity matrices for all N units: (N x d x d)
        I = torch.eye(self.d, device=self.mu.device).unsqueeze(0).expand(self.n, -1, -1)
        
        # Baseline part: multiply each identity by its unit's lambda
        W = self.lambda_base.view(self.n, 1, 1) * I
        
        # Low-rank part: Sum omega_nk * outer(v_nk, v_nk)
        # n: unit index, k: rank index, i/j: dim indices
        if self.k>0:
            low_rank_term = torch.einsum('nki,nkj->nij', self.v, self.v)
            return (W) + low_rank_term
        else:
            return W

    def forward(self, x):
        """
        x: (batch, d)
        returns: (batch, n)
        """
        # 1. Compute displacements for all pairs: (batch, n, d)
        # x is (B, 1, d), mu is (1, N, d) -> diff is (B, N, d)
        x = x.squeeze()
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
        
        # 2. Get all Hessians: (N, d, d)
        W = self.get_hessians()
        
        # 3. Compute the quadratic form: (batch-i, unit-n)^T * W_n * (batch-i, unit-n)
        # b: batch, n: unit, i: d, j: d
        # We want a result of shape (b, n)
        quad_form = torch.einsum('bni,nij,bnj->bn', diff, W, diff)
        
        # 4. Activation
        if self.type == 'no_exp':
            return -quad_form 
        # print(self.lambda_base.mean())
        return torch.exp(-quad_form / self.d)

    def extra_repr(self):
        return f'in_features={self.d}, out_features={self.n}, rank_k={self.k}'



class ResHMUMLP_probing(nn.Module):
    def __init__(self, input_channels, block, num_blocks,
                 num_slices=1, degree=1, normalize=True, madden=128,
                 num_classes=10, use_dropout=True):

        super().__init__()
        self.in_planes = madden

        # GMU layers
        self.hmu1 = HMULayer(input_channels, madden, num_slices)
       
        self.bn1 = nn.BatchNorm1d(madden)

        self.layer1 = self._make_layer(block, madden, num_blocks[0],k=num_slices)
        self.layer2 = self._make_layer(block, madden, num_blocks[1],k=num_slices)

        # Probes
        self.probe1 = nn.Linear(madden, num_classes)
        self.probe2 = nn.Linear(madden, num_classes)
        self.probe3 = nn.Linear(madden, num_classes)

        self.linear = nn.Linear(madden, num_classes)
        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0)

    def _make_layer(self, block, planes, num_blocks, stride=1,k = 1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.hmu1(x)

        out = self.bn1(x)  # x_errs is already (B,128,1,1)  # BN2d expects 4D
        out = out.squeeze(-1).squeeze(-1)

        feat1 = out  # already flat

        out = self.layer1(out.unsqueeze(-1).unsqueeze(-1))
        out = out.squeeze(-1).squeeze(-1)
        feat2 = out

        out = self.layer2(out.unsqueeze(-1).unsqueeze(-1))
        out = out.squeeze(-1).squeeze(-1)
        feat3 = out

        if self.use_dropout:
            out = self.drop(out)

        out = self.linear(out)

        return out, (feat1, feat2, feat3)


class SimpleHMUMLP(nn.Module):
    def __init__(self, input_channels,num_slices = 1, degree = 1, normalize = True, epsilon=0.0001, num_classes=10,use_dropout=True):
        super(SimpleHMUMLP, self).__init__()
        madden = 128 
        
        self.hmu1 = HMULayer(input_channels, 128, num_slices)
        
        self.use_dropout = use_dropout
        # self.recon1 = nn.Linear(madden, madden)
        self.bn1 = nn.BatchNorm2d(madden)
        self.bn2 = nn.BatchNorm1d(madden)
        self.fc = nn.Linear(madden, madden)
        
        self.linear = nn.Linear(madden, num_classes)
        
        self.drop = nn.Dropout(0)

    def forward(self, x,p=1):
        # x = self.conv1(x)
        x = self.hmu1(x)
#         print(x_errs1.shape)
        out = self.bn1(x)
        if self.use_dropout:
            out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        feats = out.view(out.size(0), -1)
        # out = self.linear(F.relu(self.bn2(self.fc(feats))))
        out = self.linear(feats)
        
        return out


class SimpleHMUMLP_probing(nn.Module):
    def __init__(self, input_channels, num_slices=1, num_classes=10, madden=128, use_dropout=True, relu= True):
        super(SimpleHMUMLP_probing, self).__init__()
        
        self.relu = relu
        # 1. HMU Layer (Geometric Feature Extraction)
        self.hmu1 = HMULayer(input_channels, madden, num_slices)
        self.bn1 = nn.BatchNorm1d(madden)
        
        # 2. Standard MLP Layer
        self.fc = nn.Linear(madden, madden)
        self.bn2 = nn.BatchNorm1d(madden)
        
        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)

        # 4. Probes (mapping intermediate dimensions to num_classes)
        self.probe1 = nn.Linear(madden, num_classes) # After HMU1
        self.probe2 = nn.Linear(madden, num_classes) # After FC1
        self.probe3 = nn.Linear(madden, num_classes) # Pre-final output

        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0) # Set to non-zero if dropout is desired

    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.hmu1(x)
        if self.relu:
            out = F.relu(self.bn1(out))
        else:
            out = self.bn1(out)

        
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
        # --- Stage 2: MLP Hidden ---
        # out = self.fc(feat1)
        # out = self.bn2(out)
        # out = F.relu(out)
        
        # if self.use_dropout:
        #     out = self.drop(out)
            
        # feat2 = out 
        
        # --- Stage 3: Output ---
        # In this simple architecture, feat3 matches feat2
        # logits = self.linear(feat2)
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, (feat1, feat1, feat1)
    

class SimpleHMUMLP2_synthetic(nn.Module):
    def __init__(self, input_channels, num_slices=1, num_classes=10, madden=128, use_dropout=True, relu= True):
        super(SimpleHMUMLP2_synthetic, self).__init__()
        
        self.relu = relu
        # 1. HMU Layer (Geometric Feature Extraction)
        self.hmu1 = HMULayer(input_channels, madden, num_slices)
        self.bn1 = nn.BatchNorm1d(madden)
        
        
        self.fc = nn.Linear(madden, madden)
        self.bn2 = nn.BatchNorm1d(madden)
        # 2. Standard MLP Layer
        # self.fc = nn.Linear(madden, madden)
        # self.bn2 = nn.BatchNorm1d(madden)
        
        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)


    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.hmu1(x)
        out = self.bn1(out)

        out = F.relu(self.bn2(self.fc(out)))
# 
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
     
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, 0


class SimpleHMUMLP_synthetic(nn.Module):
    def __init__(self, input_channels, num_slices=1, num_classes=10, madden=128, use_dropout=True, relu= True):
        super(SimpleHMUMLP_synthetic, self).__init__()
        
        self.relu = relu
        # 1. HMU Layer (Geometric Feature Extraction)
        self.hmu1 = HMULayer(input_channels, madden, num_slices)
        self.bn1 = nn.BatchNorm1d(madden)
        
        
        # 2. Standard MLP Layer
        # self.fc = nn.Linear(madden, madden)
        # self.bn2 = nn.BatchNorm1d(madden)
        
        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)


    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.hmu1(x)
        out = F.relu(self.bn1(out))
# 
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
     
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, 0
    
    
class SimpleMLP_synthetic(nn.Module):
    def __init__(self, input_channels, num_classes=10, madden=128,  use_dropout=True):
        super(SimpleMLP_synthetic, self).__init__()
        
        # 1. HMU Layer (Geometric Feature Extraction)
        self.layer1 =  nn.Linear(input_channels, madden)
        self.bn1 = nn.BatchNorm1d(madden)
        
        
        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)


    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.layer1(x)
        out = F.relu(self.bn1(out))
        
        
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
        
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, 0

class SimpleMLP2_synthetic(nn.Module):
    def __init__(self, input_channels, num_classes=10, madden=128,  use_dropout=True):
        super(SimpleMLP2_synthetic, self).__init__()
        
        # 1. HMU Layer (Geometric Feature Extraction)
        self.layer1 =  nn.Linear(input_channels, madden)
        self.bn1 = nn.BatchNorm1d(madden)

        self.fc = nn.Linear(madden, madden)
        self.bn2 = nn.BatchNorm1d(madden)

        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)


    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.layer1(x)
        out = F.relu(self.bn1(out))
        
        out = F.relu(self.bn2(self.fc(out)))
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
        
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, 0
    
    
class SimpleMLP_probing(nn.Module):
    def __init__(self, input_channels, num_classes=10, madden=128,  use_dropout=True):
        super(SimpleMLP_probing, self).__init__()
        
        # 1. HMU Layer (Geometric Feature Extraction)
        self.layer1 =  nn.Linear(input_channels, madden)
        self.bn1 = nn.BatchNorm1d(madden)
        
        # 2. Standard MLP Layer
        self.fc = nn.Linear(madden, madden)
        self.bn2 = nn.BatchNorm1d(madden)
        
        # 3. Final Classifier
        self.linear = nn.Linear(madden, num_classes)

        # 4. Probes (mapping intermediate dimensions to num_classes)
        self.probe1 = nn.Linear(madden, num_classes) # After HMU1
        self.probe2 = nn.Linear(madden, num_classes) # After FC1
        self.probe3 = nn.Linear(madden, num_classes) # Pre-final output

        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0) # Set to non-zero if dropout is desired

    def forward(self, x):
        # --- Stage 1: HMU ---
        x = x.squeeze()
        out = self.layer1(x)
        out = F.relu(self.bn1(out))
        
        # Flatten to (Batch, 128)
        feat1 = out.view(out.size(0), -1) 
        
        
        logits = self.linear(feat1)

        # Return final logits and the three probing features
        return logits, (feat1, feat1, feat1)
    


class PiNetLayer(nn.Module):
    """
    A SOTA-style Recursive Multiplicative Block (Pi-Net).
    Ref: Chrysos et al. (2023), 'Regularization of Polynomial Networks'
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.U = nn.Linear(input_dim, hidden_dim) # For multiplicative branch
        self.V = nn.Linear(input_dim, hidden_dim) # For additive branch
        self.norm = nn.LayerNorm(hidden_dim)      # Crucial for stable high-order training
        
    def forward(self, x, y_prev):
        # y_prev is the representation from the previous 'degree'
        interaction = self.U(x) * y_prev  # Multiplicative interaction
        skip = self.V(x)                  # Standard linear skip
        return self.norm(interaction + skip)

class SOTAPolynomialNetwork(nn.Module):
    def __init__(self, input_channels, num_slices=2, num_classes=2, madden=128):
        super().__init__()
        self.degree = num_slices
        # Initialization layer
        self.first_layer = nn.Linear(input_channels, madden)
        
        # Recursive polynomial layers
        self.poly_layers = nn.ModuleList([
            PiNetLayer(input_channels, madden) for _ in range(self.degree - 1)
        ])
        
        self.classifier = nn.Linear(madden, num_classes)

    def forward(self, x):
        # Start with degree 1
        x = x.squeeze()
        y = self.first_layer(x)
        feat1 = y.view(y.size(0), -1)  
        # Iteratively increase polynomial degree
        for layer in self.poly_layers:
            y = layer(x, y)
            
        logits = self.classifier(y)
        return logits, (feat1, feat1, feat1)
    
class HMUStack(nn.Module):
    def __init__(self, input_channels,num_slices = 1, degree = 1, normalize = True, epsilon=0.0001, num_classes=10,use_dropout=True, relu=True):
        super(HMUStack, self).__init__()
        madden = 128 
        # self.conv1 = nn.Conv2d(input_channels, projections, 1)
        # self.gmu1 = MaskGMULayer(input_channels, madden,epsilon = epsilon,num_slices = num_slices,degree=degree,normalize=normalize)
        # print('abbaba')
        self.hmu1 = HMULayer(input_channels, 128, num_slices)
        
        self.use_dropout = use_dropout
        # self.recon1 = nn.Linear(madden, madden)
        self.bn1 = nn.BatchNorm2d(madden)
        self.bn2 = nn.BatchNorm1d(madden)
        self.hmu2 = HMULayer(128, 128, num_slices)
        
        self.linear = nn.Linear(madden, num_classes)
        
        
        
        self.drop = nn.Dropout(0)

    def forward(self, x,p=1):
        # x = self.conv1(x)
        x = self.hmu1(x)
#         print(x_errs1.shape)
        out = self.bn1(x)
        if self.use_dropout:
            out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        feats = out.view(out.size(0), -1)
        out = self.linear(F.relu(self.bn2(self.fc(feats))))
        
        return out



class HMUStack_probing(nn.Module):
    def __init__(self, input_channels, num_slices=1, num_classes=10, madden=128, use_dropout=True,relu=True):
        super(HMUStack_probing, self).__init__()
        
        self.relu = relu
        # Layer 1
        self.hmu1 = HMULayer(input_channels, madden, num_slices,Type = 'no_exp')
        self.bn1 = nn.BatchNorm1d(madden)
        
        # Layer 2
        self.hmu2 = HMULayer(madden, madden, num_slices)
        
        self.hmu3 = HMULayer(madden, num_classes, num_slices,Type = 'no_exp')
        
        self.bn2 = nn.BatchNorm1d(madden)
        self.bn3 = nn.BatchNorm1d(num_classes)

        # Probes - mapping intermediate features to class space
        self.probe1 = nn.Linear(madden, num_classes) # After HMU1
        self.probe2 = nn.Linear(madden, num_classes) # After HMU2
        
        # Final Classifier
        self.linear = nn.Linear(madden, num_classes)
        
        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0) # Changed from 0 to 0.2 for actual effect if enabled

    def forward(self, x):
        # --- Block 1 ---
        out = self.hmu1(x)
        out = self.bn1(out)
        # Assuming HMULayer output is (B, C, 1, 1), squeeze for linear layers
        feat1 = out.view(out.size(0), -1) 
        
        # --- Block 2 ---
        # Re-shape for next HMU if it expects 4D or 2D (adjust accordingly)
        # If HMU2 expects (B, C), use feat1. If it expects (B, C, 1, 1), use out.
        out = self.hmu2(feat1)
        # out = self.bn3(out)
        
        # return out, (feat1, feat1, feat1)
        if self.relu:
            out = F.relu(self.bn2(out))
        else:
            out = self.bn2(out)


        feat2 = out.view(out.size(0), -1)

        # # --- Final Stage ---
        # if self.use_dropout:
        #     final_feat = self.drop(feat2)
        # else:
        #     final_feat = feat2
            
        logits = self.linear(feat2)

        # Return the final output and the tuple of features for probing
        # We provide feat1 and feat2 as the two major "hidden" states
        return logits, (feat1, feat2, feat2)
    
    
import torch




class GeneralHMUStack(nn.Module):
    def __init__(self, input_channels, num_layers=3, num_slices=1, num_classes=10, madden=128):
        super(GeneralHMUStack, self).__init__()
        
        self.num_layers = num_layers
        self.hmus = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer takes input_channels, subsequent layers take madden
            in_dim = input_channels if i == 0 else madden
            
            # The final HMU layer has no Type (defaults to 'exp' behavior)
            # All preceding layers are 'no_exp'
            hmu_type = 'no_exp' if i < (num_layers - 1) else None
            
            self.hmus.append(HMULayer(in_dim, madden, num_slices, Type=hmu_type))
            self.bns.append(nn.BatchNorm1d(madden))
        
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(madden, num_classes)

    def forward(self, x):
        intermediate_features = []
        out = x

        for i in range(self.num_layers):
            out = self.hmus[i](out)
            
            # Ensure output is flattened for BatchNorm1d/Linear if it comes out 4D
            if out.dim() > 2:
                out = out.view(out.size(0), -1)
            
            out = self.bns[i](out)
            # Store features for probing/analysis
            intermediate_features.append(out)
            # Apply ReLU only after the very last HMU layer per your requirement
            # if i == self.num_layers - 1:
        out = self.relu(out)
            
            

        # Final Classification Head
        logits = self.classifier(out)

        return logits, tuple(intermediate_features)
    
    
   