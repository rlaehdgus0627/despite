import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer_v1 import *



from pst_convolutions import PSTConv
from src.models.motion_clip import Encoder_TRANSFORMER


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    # axis_angle: (..., 3)
    angle = torch.norm(axis_angle, dim=-1)  # (...,)
    axis = axis_angle / (angle[..., None] + 1e-8)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(axis.shape[:-1] + (3, 3))
    cos = torch.cos(angle)[..., None, None]
    sin = torch.sin(angle)[..., None, None]
    outer = axis[..., :, None] * axis[..., None, :]
    rot = cos * eye + (1.0 - cos) * outer + sin * K
    return rot


def _axis_angle_to_rot6d(axis_angle: torch.Tensor) -> torch.Tensor:
    # axis_angle: (..., 3) -> rot6d: (..., 6)
    rot = _axis_angle_to_matrix(axis_angle)
    rot6d = rot[..., :2].reshape(axis_angle.shape[:-1] + (6,))
    return rot6d


class SMPLPoseEncoder(nn.Module):
    def __init__(self, embed_dim, n_joints, device="cuda"):
        super().__init__()
        parameters = {
            "cuda": True,
            "device": 0,
            "modelname": "motionclip_transformer_rc_rcxyz_vel",
            "latent_dim": embed_dim,
            "num_layers": 8,
            "activation": "gelu",
            "modeltype": "motionclip",
            "nfeats": 6,
            "njoints": n_joints,
        }
        self.encoder = Encoder_TRANSFORMER(**parameters).to(device)

    def forward(self, x: torch.Tensor):
        # x: (B, T, J, 3) axis-angle
        rot6d = _axis_angle_to_rot6d(x)
        return self.encoder(rot6d)


class _TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class SMPLPoseEncoderTCN(nn.Module):
    def __init__(self, embed_dim: int, n_joints: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = n_joints * 6
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _TemporalConvBlock(hidden_dim, dilation=1, dropout=dropout),
            _TemporalConvBlock(hidden_dim, dilation=2, dropout=dropout),
            _TemporalConvBlock(hidden_dim, dilation=4, dropout=dropout),
            _TemporalConvBlock(hidden_dim, dilation=8, dropout=dropout),
        )
        self.out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, J, 3) axis-angle
        rot6d = _axis_angle_to_rot6d(x).reshape(x.shape[0], x.shape[1], -1)  # (B, T, J*6)
        feats = rot6d.transpose(1, 2)  # (B, J*6, T)
        feats = self.input_proj(feats)
        feats = self.blocks(feats)
        pooled = feats.mean(dim=-1)  # (B, C)
        return {"mu": self.out(pooled)}


class _ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.ln(x).transpose(1, 2)  # (B, C, T)
        y = F.glu(self.pw1(y), dim=1)
        y = self.dw(y)
        y = self.act(self.bn(y))
        y = self.pw2(y)
        y = self.drop(y).transpose(1, 2)
        return y


class _ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        ff_dim = d_model * ff_mult
        self.ff1_ln = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = _ConformerConvModule(d_model=d_model, dropout=dropout)
        self.ff2_ln = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(self.ff1_ln(x))
        attn_in = self.attn_ln(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.attn_drop(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(self.ff2_ln(x))
        return self.final_ln(x)


class SMPLPoseEncoderConformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_joints: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = n_joints * 6
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [_ConformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, J, 3) axis-angle
        seq = _axis_angle_to_rot6d(x).reshape(x.shape[0], x.shape[1], -1)  # (B, T, J*6)
        seq = self.input_proj(seq)
        for block in self.blocks:
            seq = block(seq)
        pooled = seq.mean(dim=1)  # (B, C)
        return {"mu": self.out(pooled)}


class _GraphConvTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_t: int = 3, dropout: float = 0.1):
        super().__init__()
        pad_t = kernel_size_t // 2
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size_t, 1), padding=(pad_t, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J), adj: (J, J)
        y = self.gcn(x)
        y = torch.einsum("bctj,jk->bctk", y, adj)
        y = self.tcn(y)
        return self.act(y + self.residual(x))


def _build_smpl_adjacency(n_joints: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # SMPL-like tree (falls back gracefully for truncated joint sets).
    parent = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
    ]
    adj = torch.eye(n_joints, device=device, dtype=dtype)
    for child in range(min(n_joints, len(parent))):
        p = parent[child]
        if p >= 0 and p < n_joints:
            adj[child, p] = 1.0
            adj[p, child] = 1.0
    if n_joints > len(parent):
        for j in range(len(parent), n_joints - 1):
            adj[j, j + 1] = 1.0
            adj[j + 1, j] = 1.0
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / deg


class SMPLPoseEncoderSTGCN(nn.Module):
    def __init__(self, embed_dim: int, n_joints: int, in_channels: int = 6, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_joints = n_joints
        self.block1 = _GraphConvTemporalBlock(in_channels, hidden_dim, dropout=dropout)
        self.block2 = _GraphConvTemporalBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.block3 = _GraphConvTemporalBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, J, 3) axis-angle
        rot6d = _axis_angle_to_rot6d(x)  # (B, T, J, 6)
        feats = rot6d.permute(0, 3, 1, 2).contiguous()  # (B, 6, T, J)
        adj = _build_smpl_adjacency(self.n_joints, feats.device, feats.dtype)
        feats = self.block1(feats, adj)
        feats = self.block2(feats, adj)
        feats = self.block3(feats, adj)
        pooled = feats.mean(dim=(2, 3))  # global temporal + joint pool
        return {"mu": self.out(pooled)}

class MSRAction(nn.Module):
    def __init__(self, radius=1.5, nsamples=3*3, num_classes=20):
        super(MSRAction, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=284,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out

class NTU(nn.Module):
    def __init__(self, radius=0.1, nsamples=3*3, num_classes=20):
        super(NTU, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[0,0])

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[0,0])

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out

class IMUEncoder(torch.nn.Module):
    def __init__(self, input_size=15, hidden_size=30, num_layers=2, device="cuda") -> None:
        super(IMUEncoder, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        #self.imu_bgru = nn.GRU(input_size=lstm_input_size, hidden_size=self.hidden_size, num_layers=lstm_num_layers, bidirectional=False, batch_first=True)
        
    def forward(self, x):
        """
        x: (Batch_size, sequence_length, input_size)
        """
           
        # hn: (num_layers * num_directions, batch_size, hidden_size)     
        # output, (hn, cn) = self.imu_lstm(x)
        # output, hn = self.imu_bgru(x)
        
        # # output = output[:, -1, :]
        
        # result = output[:, -1, :]
        
        # return result
        
        B, L, H = x.size()
        
        h_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        embeddings = out[:, -1, :].reshape(-1, self.hidden_size)
        
        return embeddings


class PSTTransformerLegacy(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        features = features.permute(0, 1, 3, 2)

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output


class PSTTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        #self.mlp_head = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, mlp_dim),
        #    nn.GELU(),
        #    nn.Dropout(dropout2),
        #    nn.Linear(mlp_dim, num_classes),
        #)

        ### SupCon/SimClr projection head
        self.head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, mlp_dim)
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        features = features.permute(0, 1, 3, 2)

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.head(output)
        #output = F.normalize(output, dim=1)  ### Normalize unitsphere
        return output
