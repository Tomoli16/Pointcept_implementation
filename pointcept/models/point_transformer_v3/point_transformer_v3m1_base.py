"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "modules", "mamba"))

from mamba_ssm.modules.mamba_simple import Mamba  # oder was du brauchst

from enum import Enum

class BlockType(str, Enum):
    CONV = "conv"
    MAMBA = "mamba"
    ATTENTION = "attention"

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    # Bringt die Punktzahl jedes Punktwolken-Batches auf ein Vielfaches von patch_size
    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            # Anzahl der Punkte pro Batch
            bincount = offset2bincount(offset)
            # Länge wird auf das nächste Vielfache von patch_size aufgerundet
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            # Startpositionen der Batches einmal mit einmal ohne Padding
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            # Aufzählung von 1 bis zur Gesamtzahl alles gepaddeten Punkte bzw
            # aller originalen Punkte
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            # Cumulative Sequence Lengths
            cu_seqlens = []
            for i in range(len(offset)):
                # Verschiebt die Indizes der originalen Punkte auf die Position an der
                # sie sich im gepaddeten Tensor befinden würden
                # Eine Art forward mapping um später reverse mapping zu machen
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                # bincount[i] Originalanzahl einer Sezene
                # bincount_pad[i] Anzahl der gepaddeten Punkte

                # Ziel: Leere Slots am Ende die durch Padding entstanden sind
                # werden mit echten Punkten gefüllt (Kopien)
                if bincount[i] != bincount_pad[i]:
                    # Zielbereich, die zu füllenden Stellen
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[    # Quellbereich, die kopiert werden, genau die letzen Punkte aus 
                        _offset_pad[i + 1]                      # dem letzen vollständigen Patch
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                # Am Ende soll pad nur gültige Indizes enthalten, verschiebe also zurück 
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                # Erzeugt pro Szene Start-Offsets für alle Patches
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            # cu_seqlens = [0, 64, 128, 192, 256] (Beispiel)
            # Genau das erwartet Flash Attention
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            # Wähle Patchsize so, dass sie nicht größer ist als das kleinste Batch
            # und nicht größer als die maximale Patchgröße
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # Die Positionen der (gepadde­ten und ggf. duplizierten) Punkte, 
        # aber sortiert nach der gewünschten Reihenfolge (z. B. Z-Order, Hilbert etc.)
        # Länge N_padded
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        # feat: (N, C) -> (N_padded, C*3) Linear Layer
        # Wandelt Features in QKV um (linear Layer) und sortiert sie anschließend
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                # Macht aus einem flachen, aneinander gereihten Q, K, V Vektor (Je Token)
                # eine strukturierte Darstellung die getrennte Q, K, V Matrizen enthält,
                # in Heads aufgeteilt ist als float16 vorliegt
                # [N', C*3] -> [N', 3, H, C//H]
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)    # Wandelt wieder in flachen Vektor um [num_points, C]
            # Rechnet zurück zu float32
            feat = feat.to(qkv.dtype)
        # Punkte werden in original Reihenfolge zurücksortiert
        # feat: (N', C) -> (N, C)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point
    
class MambaSerialized(PointModule):
    def __init__(self, channels, patch_size, order_index, proj_drop=0.0):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.order_index = order_index
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.patch_size_max = patch_size

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            # Anzahl der Punkte pro Batch
            bincount = offset2bincount(offset)
            # Länge wird auf das nächste Vielfache von patch_size aufgerundet
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            # Startpositionen der Batches einmal mit einmal ohne Padding
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            # Aufzählung von 1 bis zur Gesamtzahl alles gepaddeten Punkte bzw
            # aller originalen Punkte
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            # Cumulative Sequence Lengths
            cu_seqlens = []
            for i in range(len(offset)):
                # Verschiebt die Indizes der originalen Punkte auf die Position an der
                # sie sich im gepaddeten Tensor befinden würden
                # Eine Art forward mapping um später reverse mapping zu machen
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                # bincount[i] Originalanzahl einer Sezene
                # bincount_pad[i] Anzahl der gepaddeten Punkte

                # Ziel: Leere Slots am Ende die durch Padding entstanden sind
                # werden mit echten Punkten gefüllt (Kopien)
                if bincount[i] != bincount_pad[i]:
                    # Zielbereich, die zu füllenden Stellen
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[    # Quellbereich, die kopiert werden, genau die letzen Punkte aus 
                        _offset_pad[i + 1]                      # dem letzen vollständigen Patch
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                # Am Ende soll pad nur gültige Indizes enthalten, verschiebe also zurück 
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                # Erzeugt pro Szene Start-Offsets für alle Patches
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            # cu_seqlens = [0, 64, 128, 192, 256] (Beispiel)
            # Genau das erwartet Flash Attention
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):

        self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        # print("Feat shape before:", point.feat.shape)
        # print("Padding shape", pad.shape)
        feat = point.feat[order]  # (N_padded, C)
        # print("order.shape:", order.shape)
        # print("point.feat.shape:", feat.shape)
        feat = feat.reshape(-1, self.patch_size, self.channels)  # [B, K, C]
        feat = self.norm(feat)
        feat = self.mamba(feat)  # MambaBlock läuft komplett in Python
        feat = feat.reshape(-1, self.channels)[inverse]  # [N, C]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point



class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class simpleBlock(PointModule):
    def __init__(
        self,
        channels,
        norm_layer=nn.LayerNorm,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        pre_norm=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.channels = channels
        self.norm = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
            )
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.norm(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        return point
    
class simpleBlockConv(PointModule):
    def __init__(
        self,
        channels,
        norm_layer=nn.LayerNorm,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        pre_norm=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.channels = channels
        self.norm = PointSequential(norm_layer(channels))
        self.conv = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="simple_block"
            )
        )

        self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.norm(point)
        point = self.conv(point)
        point = self.act(point)
        point.feat = shortcut + point.feat
        return point



class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # Convolutional Position Encoding
        # Reichert Punktfeatures mit lokalem Kontext an
        # Berücksichtigt räumliche Struktur
        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,       # in channels
                channels,       # out channels
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            # positionssensitiv, weil das Input-Feature schon lokalen Kontext enthält
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        # Eine Art Dropout aber nicht auf einzelne Neuronen
        # sondern auf ganze Pfade
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        # Entscheidet ob Ergebnis von Attention oder 0 genutzt wird
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point
    

class MambaBlock(PointModule):
    def __init__(
        self,
        channels,
        patch_size=48,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,       # in channels
                channels,       # out channels
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            # positionssensitiv, weil das Input-Feature schon lokalen Kontext enthält
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.mamba = MambaSerialized(
            channels=channels,
            patch_size=patch_size,
            order_index=order_index,
            proj_drop=proj_drop,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )

        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        # Entscheidet ob Ergebnis von Attention oder 0 genutzt wird
        # point = self.drop_path(self.mamba(point))
        point = self.mamba(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point




class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Prüft ob stride eine Potenz von 2 ist
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        # Bestimmt wie viele Bit-Ebenen nach oben gegangen werden soll
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        # Schneidet die letzten (feinsten) pooling_depth * 3 Bits ab
        # Das Bit-Shift in code = point.serialized_code >> (pooling_depth * 3) 
        # dient dazu, die Punktwolke auf einer gröberen Auflösung zu betrachten – 
        # also die Punkte in größere "Voxel-Cluster" zu gruppieren
        code = point.serialized_code >> pooling_depth * 3
        # Gibt unique codes
        # Für jeden Index Clusterzugehörigkeit
        # und Anzahl der Punkte in jedem Cluster
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        # cumsum gibt jeweils die Endindizes der Cluster an (Nicht inklusive)
        # Vorne wird noch 0 hinzugefügt
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        # head_indices enthält jeweils den ersten Index jedes Clusters im sortierten Punkt-Array
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        # rediziert Code auf die repräsentativen Punkte
        # [num_orders, num_points] -> [num_orders, num_clusters]
        code = code[:, head_indices]
        # Sortiert nach Code
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            # Hier passiert das Pooling
            # feat bekommt durch Linear proj mehr channels
            # Dann wird nach indices sortiert
            # idx_ptr gibt dann Clusterabschnitte an
            # torch reduziert dann
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        # PointSequential ist ein Container sowie nn.Sequential, nur für Punktwolken
        self.stem = PointSequential(
            # SubMConv3d ist eine 3D-Convolution, die Sparse Convolution verwendet
            # Submanifold Convolution, d.h. Output Koordinaten sind die gleichen wie Input Koordinaten
            # und die Anzahl der Kanäle wird geändert
            # Conv wird also mit jedem Punkt als Zentrum durchgeführt
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,  # 5x5x5 Kernel
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        block_type="attention",
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.block_type = block_type

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) +1
        assert self.cls_mode or self.num_stages == len(dec_channels) +1
        assert self.cls_mode or self.num_stages == len(dec_num_head) +1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) +1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder wird im Konstruktor gebaut
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    # Num Points sinkt aber channels steigen
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                name = f"block{i}"
                if block_type == BlockType.CONV:
                    enc.add(
                        simpleBlockConv(
                            channels=enc_channels[s],
                            norm_layer=ln_layer,
                            mlp_ratio=mlp_ratio,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                        ),
                        name=name,
                    )
                elif block_type == BlockType.MAMBA:
                    enc.add(
                        MambaBlock(
                            channels=enc_channels[s],
                            patch_size=enc_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            proj_drop=proj_drop,
                            drop_path=enc_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                        ),
                        name=name,
                    )
                elif block_type == BlockType.ATTENTION:
                    enc.add(
                        Block(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            patch_size=enc_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=enc_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=name,
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")
            
            # for i in range(enc_depths[s]):
                
            #     enc.add(
            #         simpleBlockConv(
            #             channels=enc_channels[s],
            #             norm_layer=ln_layer,
            #             mlp_ratio=mlp_ratio,
            #             act_layer=act_layer,
            #             pre_norm=pre_norm,
            #         ),
            #         name=f"block{i}",
            #     )
                # enc.add(
                #     MambaBlock(
                #         channels=enc_channels[s],
                #         patch_size=enc_patch_size[s],
                #         mlp_ratio=mlp_ratio,
                #         proj_drop=proj_drop,
                #         drop_path=enc_drop_path_[i],
                #         norm_layer=ln_layer,
                #         act_layer=act_layer,
                #         pre_norm=pre_norm,
                #         order_index=i % len(self.order),
                #         cpe_indice_key=f"stage{s}",
                #     ),
                #     name=f"block{i}",
                # )
                # enc.add(
                #     Block(
                #         channels=enc_channels[s],
                #         num_heads=enc_num_head[s],
                #         patch_size=enc_patch_size[s],
                #         mlp_ratio=mlp_ratio,
                #         qkv_bias=qkv_bias,
                #         qk_scale=qk_scale,
                #         attn_drop=attn_drop,
                #         proj_drop=proj_drop,
                #         drop_path=enc_drop_path_[i],
                #         norm_layer=ln_layer,
                #         act_layer=act_layer,
                #         pre_norm=pre_norm,
                #         order_index=i % len(self.order),
                #         cpe_indice_key=f"stage{s}",
                #         enable_rpe=enable_rpe,
                #         enable_flash=enable_flash,
                #         upcast_attention=upcast_attention,
                #         upcast_softmax=upcast_softmax,
                #     ),
                #     name=f"block{i}",
                # )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        # Nur wenn wir nicht im klassifizierungsmodus sind, sonst reicht latente Repräsentation
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )

                for i in range(dec_depths[s]):
                    name = f"block{i}"
                    if self.block_type == BlockType.CONV:
                        dec.add(
                            simpleBlockConv(
                                channels=dec_channels[s],
                                norm_layer=ln_layer,
                                mlp_ratio=mlp_ratio,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                            ),
                            name=name,
                        )
                    elif self.block_type == BlockType.MAMBA:
                        dec.add(
                            MambaBlock(
                                channels=dec_channels[s],
                                patch_size=dec_patch_size[s],
                                mlp_ratio=mlp_ratio,
                                proj_drop=proj_drop,
                                drop_path=dec_drop_path_[i],
                                norm_layer=ln_layer,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                                order_index=i % len(self.order),
                                cpe_indice_key=f"stage{s}",
                            ),
                            name=name,
                        )
                    elif self.block_type == BlockType.ATTENTION:
                        dec.add(
                            Block(
                                channels=dec_channels[s],
                                num_heads=dec_num_head[s],
                                patch_size=dec_patch_size[s],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop,
                                proj_drop=proj_drop,
                                drop_path=dec_drop_path_[i],
                                norm_layer=ln_layer,
                                act_layer=act_layer,
                                pre_norm=pre_norm,
                                order_index=i % len(self.order),
                                cpe_indice_key=f"stage{s}",
                                enable_rpe=enable_rpe,
                                enable_flash=enable_flash,
                                upcast_attention=upcast_attention,
                                upcast_softmax=upcast_softmax,
                            ),
                            name=name,
                        )
                    else:
                        raise ValueError(f"Unknown block type: {self.block_type}")

                # for i in range(dec_depths[s]):
                #     dec.add(
                #         simpleBlockConv(
                #             channels=dec_channels[s],
                #             norm_layer=ln_layer,
                #             mlp_ratio=mlp_ratio,
                #             act_layer=act_layer,
                #             pre_norm=pre_norm,
                #         ),
                #         name=f"block{i}",
                #     )
                    # dec.add(
                    #     MambaBlock(
                    #         channels=dec_channels[s],
                    #         patch_size=dec_patch_size[s],
                    #         mlp_ratio=mlp_ratio,
                    #         proj_drop=proj_drop,
                    #         drop_path=dec_drop_path_[i],
                    #         norm_layer=ln_layer,
                    #         act_layer=act_layer,
                    #         pre_norm=pre_norm,
                    #         order_index=i % len(self.order),
                    #         cpe_indice_key=f"stage{s}",
                    #     ),
                    #     name=f"block{i}",
                    # )
                    # dec.add(
                    #     Block(
                    #         channels=dec_channels[s],
                    #         num_heads=dec_num_head[s],
                    #         patch_size=dec_patch_size[s],
                    #         mlp_ratio=mlp_ratio,
                    #         qkv_bias=qkv_bias,
                    #         qk_scale=qk_scale,
                    #         attn_drop=attn_drop,
                    #         proj_drop=proj_drop,
                    #         drop_path=dec_drop_path_[i],
                    #         norm_layer=ln_layer,
                    #         act_layer=act_layer,
                    #         pre_norm=pre_norm,
                    #         order_index=i % len(self.order),
                    #         cpe_indice_key=f"stage{s}",
                    #         enable_rpe=enable_rpe,
                    #         enable_flash=enable_flash,
                    #         upcast_attention=upcast_attention,
                    #         upcast_softmax=upcast_softmax,
                    #     ),
                    #     name=f"block{i}",
                    # )
                self.dec.add(module=dec, name=f"dec{s}")

    # bekommt batch
    def forward(self, data_dict):
        # Erstellt addict Dict 
        point = Point(data_dict)
        # Fügt zu Point Objekt serialization code, order und inverse hinzu
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        # Fügt sparse_shape und sparse_conv_feat hinzu für SpConv
        # "Du kannst diesen Tensor nun direkt in ein Layer wie spconv.SubMConv3d(...) einspeisen"
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
