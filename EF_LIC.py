from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

G_A_CH = 368
G_S_CH = 368
G_CH_Y = 256
G_CH_Z = 128
SP_CHANNEL = 384
G_A_DEPTH = 6
G_S_DEPTH = 12
HYPER_CH = 256

N_CODEBOOKS = (1, 2, 3, 4, 5)
CODEBOOK_DIM = 8
N_E = (1024, 512, 256, 128, 1024)
IN_DIMS = (256, 256, 256, 256, 128)
SCALE_LOWER_BOUND = 0.5
DEFAULT_FORCE_IND = 2

class FastConv2d(nn.Conv2d):
    def reset_parameters(self):
        pass

class FastEmbedding(nn.Embedding):
    def reset_parameters(self):
        pass

def conv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1, bias: bool = True):
    return FastConv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias)


class ReLUChunkAdd(nn.Module):
    def forward(self, x):
        x = F.relu(x, inplace=True)
        x1, x2 = x.chunk(2, 1)
        return x1 + x2


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = False, force_adaptor: bool = False):
        super().__init__()
        self.adaptor = FastConv2d(in_ch, out_ch, 1) if (in_ch != out_ch or force_adaptor) else None
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            FastConv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
            FastConv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            FastConv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            FastConv2d(out_ch, out_ch * 4, 1),
            ReLUChunkAdd(),
            FastConv2d(out_ch * 2, out_ch, 1),
        )

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x)
        out.add_(x)
        out.add_(self.ffn(out))
        if self.shortcut:
            out.add_(x)
        return out


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(FastConv2d(in_ch, out_ch * 4, 1), nn.PixelShuffle(2))

    def forward(self, x):
        return self.conv(x)

class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = FastConv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        return self.conv(self.down(x))


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        return self.conv(self.up(x))

class IntraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_1 = nn.Sequential(FastConv2d(192, G_A_CH, 1))
        self.enc_2 = nn.Sequential(
            *[DepthConvBlock(G_A_CH, G_A_CH) for _ in range(G_A_DEPTH)],
            conv(G_A_CH, G_CH_Y, 3, stride=2),
        )

    def forward(self, x):
        return self.enc_2(self.enc_1(F.pixel_unshuffle(x, 8)))


class IntraDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(G_CH_Y, G_S_CH),
            *[DepthConvBlock(G_S_CH, G_S_CH) for _ in range(G_S_DEPTH)],
        )
        self.dec_2 = conv(G_S_CH, 192, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.dec_2(self.dec_1(x)), 8)


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            FastConv2d(G_CH_Y, HYPER_CH, 1),
            DepthConvBlock(HYPER_CH, HYPER_CH),
            ResidualBlockWithStride2(HYPER_CH, HYPER_CH),
            ResidualBlockWithStride2(HYPER_CH, G_CH_Z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(G_CH_Z, HYPER_CH),
            ResidualBlockUpsample(HYPER_CH, HYPER_CH),
            DepthConvBlock(HYPER_CH, HYPER_CH),
            conv(HYPER_CH, G_CH_Y, 1),
        )

    def forward(self, x):
        return self.conv(x)

class WNConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty(out_ch, 1, 1, 1))
        self.weight_v = nn.Parameter(torch.empty(out_ch, in_ch, 1, 1))
        self.bias = nn.Parameter(torch.empty(out_ch))

    def weight(self):
        denom = self.weight_v.square().sum(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min_(1e-12)
        return self.weight_v * (self.weight_g / denom)

    def forward(self, x):
        return F.conv2d(x, self.weight(), self.bias)

class VectorQuantizerProjInfer(nn.Module):
    def __init__(self, n_e: int, in_dim: int):
        super().__init__()
        self.n_e = n_e
        self.e_dim = CODEBOOK_DIM
        self.embedding = FastEmbedding(n_e, CODEBOOK_DIM)
        self.register_buffer("counter", torch.zeros(n_e))
        self.in_proj = WNConv2d(in_dim, CODEBOOK_DIM)
        self.out_proj = WNConv2d(CODEBOOK_DIM, in_dim)

    @torch.inference_mode()
    def init_inference_cache(self):
        emb = self.embedding.weight.detach()  # [N,8]
        self._codebook_T = emb.t().contiguous()
        self._codebook_norm_sq = emb.square().sum(dim=1).contiguous()
        w = self.out_proj.weight().flatten(1)  # [C_in,8]
        self._decode_codebook = emb.matmul(w.t())
        self._decode_codebook.add_(self.out_proj.bias)
        self._decode_codebook = self._decode_codebook.contiguous()
        self._decode_dim = self._decode_codebook.shape[1]
        return self

    def _nearest(self, z_flat: torch.Tensor) -> torch.Tensor:
        d = torch.mm(z_flat, self._codebook_T)
        d.mul_(-2.0).add_(self._codebook_norm_sq)
        return d.argmin(1)

    def _decode_flat(self, indices, out: Optional[torch.Tensor] = None):
        idx = indices.to(device=self._decode_codebook.device, dtype=torch.long, non_blocking=True).reshape(-1)
        if out is None:
            return torch.index_select(self._decode_codebook, 0, idx)
        torch.index_select(self._decode_codebook, 0, idx, out=out)
        return out

    @torch.inference_mode()
    def encode_index(self, z: torch.Tensor):
        z = self.in_proj(z)
        B, C, H, W = z.shape
        ind_flat = self._nearest(z.permute(0, 2, 3, 1).reshape(-1, C))
        return ind_flat.view(B, H, W)

    @torch.inference_mode()
    def encoding(self, z: torch.Tensor, flat_buffer: Optional[torch.Tensor] = None):
        z = self.in_proj(z)
        B, C, H, W = z.shape
        ind_flat = self._nearest(z.permute(0, 2, 3, 1).reshape(-1, C))
        flat = self._decode_flat(ind_flat, out=flat_buffer)
        return ind_flat.view(B, H, W), flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2)

    @torch.inference_mode()
    def decoding(self, indices: torch.Tensor):
        B, H, W = indices.shape
        flat = self._decode_flat(indices)
        return flat.view(B, H, W, self._decode_dim).permute(0, 3, 1, 2).contiguous()

class ResidualVectorQuantizeDropInfer(nn.Module):
    def __init__(self, n_codebooks: int, n_e: int, in_dim: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.quantizers = nn.ModuleList([VectorQuantizerProjInfer(n_e, in_dim) for _ in range(n_codebooks)])

    @torch.inference_mode()
    def build_vq_indices(self):
        for q in self.quantizers:
            q.init_inference_cache()
        return self

    @staticmethod
    def _buffer(q: VectorQuantizerProjInfer, residual: torch.Tensor, buf: Optional[torch.Tensor]):
        B, _, H, W = residual.shape
        shape = (B * H * W, q._decode_dim)
        if buf is None or buf.shape != shape or buf.device != q._decode_codebook.device or buf.dtype != q._decode_codebook.dtype:
            return q._decode_codebook.new_empty(shape)
        return buf

    @torch.inference_mode()
    def encode_decode(self, z: torch.Tensor):
        residual = z
        z_q = torch.zeros_like(z)
        packed = None
        buf = None
        last = self.n_codebooks - 1
        for i, q in enumerate(self.quantizers):
            buf = self._buffer(q, residual, buf)
            ind_i, z_q_i = q.encoding(residual, flat_buffer=buf)
            z_q.add_(z_q_i)
            if i != last:
                residual.sub_(z_q_i)
            if packed is None:
                packed = ind_i.new_empty((self.n_codebooks, *ind_i.shape))
            packed[i].copy_(ind_i)
        return packed, z_q

    @torch.inference_mode()
    def encoding(self, z: torch.Tensor):
        residual = z
        packed = None
        buf = None
        last = self.n_codebooks - 1
        for i, q in enumerate(self.quantizers):
            if i == last:
                ind_i = q.encode_index(residual)
            else:
                buf = self._buffer(q, residual, buf)
                ind_i, z_q_i = q.encoding(residual, flat_buffer=buf)
                residual.sub_(z_q_i)
            if packed is None:
                packed = ind_i.new_empty((self.n_codebooks, *ind_i.shape))
            packed[i].copy_(ind_i)
        return packed

    @torch.inference_mode()
    def decoding(self, inds: torch.Tensor):
        # inds: [K,B,H,W]
        n = inds.shape[0]
        _, B, H, W = inds.shape
        M = B * H * W
        C = self.quantizers[0]._decode_dim
        acc = self.quantizers[0]._decode_codebook.new_zeros((M, C))
        buf = acc.new_empty((M, C))
        for i in range(n):
            self.quantizers[i]._decode_flat(inds[i], out=buf)
            acc.add_(buf)
        return acc.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    
class model(nn.Module):
    _QT = (
        ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)),
        ((0, 1, 0), (1, 1, 1), (2, 0, 0), (3, 0, 1)),
        ((0, 1, 1), (1, 1, 0), (2, 0, 1), (3, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (2, 1, 1), (3, 1, 0)),
    )

    def __init__(self):
        super().__init__()
        self.g_a = IntraEncoder()
        self.g_s = IntraDecoder()
        self.h_a = HyperEncoder()
        self.h_s = HyperDecoder()

        self.quantizes = nn.ModuleList(
            nn.ModuleList(ResidualVectorQuantizeDropInfer(n_cb, n_e, in_dim) for n_e, in_dim in zip(N_E, IN_DIMS))
            for n_cb in N_CODEBOOKS
        )
        self.adaptor = nn.ModuleList(DepthConvBlock(G_CH_Y * (i + 4), SP_CHANNEL, force_adaptor=True) for i in range(4))
        self.context_predictor = nn.Sequential(
            DepthConvBlock(SP_CHANNEL, SP_CHANNEL),
            DepthConvBlock(SP_CHANNEL, SP_CHANNEL),
            conv(SP_CHANNEL, G_CH_Y * 2, 1),
        )
        
        self.n_e = N_E

    @torch.inference_mode()
    def prepare_inference_(self, force_ind: Optional[int] = DEFAULT_FORCE_IND):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        rows = range(len(self.quantizes)) if force_ind is None else (force_ind,)
        for r in rows:
            for q in self.quantizes[r]:
                q.build_vq_indices()
        return self

    def _qt_select(self, y: torch.Tensor, idx: int, out: torch.Tensor):
        C4 = y.shape[1] // 4
        for part, (g, r, c) in enumerate(self._QT[idx]):
            out[:, part * C4:(part + 1) * C4].copy_(y[:, g * C4:(g + 1) * C4, r::2, c::2])
        return out

    def _support_buf(self, support: torch.Tensor):
        B, C, H, W = support.shape
        H2, W2 = H >> 1, W >> 1
        buf = support.new_empty(B, 7 * C, H2, W2)
        for i in range(4):
            self._qt_select(support, i, buf[:, i * C:(i + 1) * C])
        return buf

    def _qt_put_(self, dst: torch.Tensor, y_i: torch.Tensor, idx: int):
        C4 = y_i.shape[1] // 4
        for part, (g, r, c) in enumerate(self._QT[idx]):
            dst[:, g * C4:(g + 1) * C4, r::2, c::2].copy_(y_i[:, part * C4:(part + 1) * C4])
        return dst

    def _mean_scale(self, support_buf: torch.Tensor, i: int):
        C = G_CH_Y
        params = self.context_predictor(self.adaptor[i](support_buf[:, :(4 + i) * C]))
        params[:, C:].clamp_min_(SCALE_LOWER_BOUND)
        return params[:, :C], params[:, C:]

    @torch.inference_mode()
    def compress(self, x: torch.Tensor, force_ind: int = DEFAULT_FORCE_IND):
        y = self.g_a(x)
        z_inds, z_hat = self.quantizes[force_ind][-1].encode_decode(self.h_a(y))
        support_buf = self._support_buf(self.h_s(z_hat))

        B, _, H2, W2 = support_buf.shape
        y_slice = y.new_empty(B, G_CH_Y, H2, W2)
        y_inds = []

        for i in range(4):
            mean, scale = self._mean_scale(support_buf, i)
            self._qt_select(y, i, y_slice)
            y_slice.sub_(mean).div_(scale)

            if i < 3:
                ind_i, y_hat_i = self.quantizes[force_ind][i].encode_decode(y_slice)
                y_hat_i.mul_(scale).add_(mean)
                support_buf[:, (4 + i) * G_CH_Y:(5 + i) * G_CH_Y].copy_(y_hat_i)
            else:
                ind_i = self.quantizes[force_ind][i].encoding(y_slice)
            y_inds.append(ind_i)

        return {"z_inds": z_inds, "y_inds": y_inds, "force_ind": force_ind}

    @torch.inference_mode()
    def decompress(self, inds, force_ind: Optional[int] = None):
        if force_ind is None:
            force_ind = int(inds.get("force_ind", DEFAULT_FORCE_IND))

        z_hat = self.quantizes[force_ind][-1].decoding(inds["z_inds"])
        support_buf = self._support_buf(self.h_s(z_hat))
        B, _, H2, W2 = support_buf.shape
        y_hat = support_buf.new_empty(B, G_CH_Y, H2 << 1, W2 << 1)

        for i, ind_i in enumerate(inds["y_inds"]):
            mean, scale = self._mean_scale(support_buf, i)
            y_i = self.quantizes[force_ind][i].decoding(ind_i)
            y_i.mul_(scale).add_(mean)
            self._qt_put_(y_hat, y_i, i)
            if i < 3:
                support_buf[:, (4 + i) * G_CH_Y:(5 + i) * G_CH_Y].copy_(y_i)

        return self.g_s(y_hat)
