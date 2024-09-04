import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ei
import lovely_tensors as lt
from einops.layers.torch import Rearrange

lt.monkey_patch()

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.sqrt_dim = np.sqrt(out_dim)
    
    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)
    
    
    def forward(self, h, e):
        
        b, q1, c = h.shape
        b, q2, q3, c = e.shape
        assert q1 == q2 == q3
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        Q_h = ei.rearrange(Q_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        K_h = ei.rearrange(K_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        V_h = ei.rearrange(V_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        proj_e = ei.rearrange(proj_e, 'B L1 L2 (H C) -> B H L1 L2 C', H=self.num_heads, C=self.out_dim)
        

        score = Q_h[:, :, :, None, :] * K_h[:, :, None, :, :] / self.sqrt_dim
        score = ei.repeat(Q_h, 'B H L C -> B H L Lr C', Lr=q1) * ei.repeat(K_h, 'B H L C -> B H Lr L C', Lr=q2) / self.sqrt_dim

        score = score * proj_e
        e_out = score
        score = ei.reduce(score, 'B H L1 L2 C -> B H L1 L2', 'sum').clamp(-5, 5)
        score = torch.nn.functional.softmax(score, dim=2)
        h_out = score @ V_h
        
        h_out = ei.rearrange(h_out, 'B H L C -> B L (H C)')
        e_out = ei.rearrange(e_out, 'B H L1 L2 C -> B L1 L2 (H C)') 
        
        return h_out, e_out


class GraphTransformerLayerDense(nn.Module):
    
    def _build_norm(self, out_dim, num_nodes):
        if self.layer_norm:
            norm_h = nn.LayerNorm(out_dim)
            norm_e = nn.LayerNorm(out_dim)
        elif self.batch_norm:
            norm_h = nn.Sequential(
                Rearrange('B L C -> B C L'),
                nn.BatchNorm1d(out_dim),
                Rearrange('B C L -> B L C')
            )
            norm_e = nn.Sequential(
                Rearrange('B L1 L2 C -> B C (L1 L2)'),
                nn.BatchNorm1d(out_dim),
                Rearrange('B C (L1 L2) -> B L1 L2 C', L1=num_nodes, L2=num_nodes)
            )
        else:
            norm_h = nn.Identity()
            norm_e = nn.Identity()
        return norm_h, norm_e
        
    
    def __init__(self, in_dim, out_dim, num_heads, num_nodes=100, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False) -> None:
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.num_nodes = num_nodes
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        
        self.norm1_h, self.norm1_e = self._build_norm(out_dim=out_dim, num_nodes=num_nodes)
            
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)
        
        self.norm2_h, self.norm2_e = self._build_norm(out_dim=out_dim, num_nodes=num_nodes)

        
    def forward(self, h, e):
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(h, e)
        
        h = h_attn_out
        e = e_attn_out
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        h = self.norm1_h(h)
        e = self.norm1_e(e)
        

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection           

        h = self.norm2_h(h)
        e = self.norm2_e(e)
        
        return h, e

if __name__ == '__main__':
    l = GraphTransformerLayerDense(256, 256, 2)
    x = torch.rand(20, 100, 256)
    e = torch.rand(20, 100, 100, 256)
    x = l(x, e)
    x = x