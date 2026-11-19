"""SUL core model with a bias-free semantic-unit output projection.

This variant follows the paper's matrix output projection. Input normalization
and the active gating, aggregation, and matching computations are retained.
"""
from contextlib import nullcontext, contextmanager
from functools import wraps
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import BertModel, BertTokenizer, ViTModel


def configure_precision(opt):
    mode = getattr(opt, 'precision', 'fp32')
    if mode == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif mode == 'bf16':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('bf16 needs supported CUDA hardware')
        torch.backends.cuda.matmul.allow_tf32 = False
    elif mode != 'fp32':
        raise ValueError('Unknown precision: ' + mode)

def precision_context(opt):
    return bf16_context() if getattr(opt, 'precision', 'fp32') == 'bf16' else nullcontext()

@contextmanager
def bf16_context():
    with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
        yield

def full_precision_context():
    return torch.autocast('cuda', enabled=False) if torch.is_autocast_enabled() else nullcontext()

def precision_forward(function):

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with precision_context(self.opt):
            return function(self, *args, **kwargs)
    return wrapped

def attention_options(opt):
    implementation = getattr(opt, 'attn_impl', None)
    return {'attn_implementation': implementation} if implementation is not None else {}

class VisionTransEncoder(nn.Module):
    """ViT backbone and projection to the shared feature dimension."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.vit_type == 'vit_384':
            self.visual_encoder = ViTModel.from_pretrained(
                'google/vit-base-patch16-384', **attention_options(opt))
            opt.num_patches = 576
        elif opt.vit_type in ('vit', 'vit_224'):
            self.visual_encoder = ViTModel.from_pretrained(
                getattr(opt, 'local_vit_path', 'google/vit-base-patch16-224-in21k'),
                **attention_options(opt))
            opt.num_patches = 196
        else:
            raise ValueError('This core implementation supports ViT-224 and ViT-384')
        if opt.embed_size == self.visual_encoder.config.hidden_size:
            self.vision_proj = nn.Identity()
        else:
            self.vision_proj = nn.Linear(self.visual_encoder.config.hidden_size, opt.embed_size)

    def forward(self, images):
        return self.vision_proj(self.visual_encoder(images).last_hidden_state)


class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()
        self.opt = opt
        self.embed_size = opt.embed_size
        self.tokenizer = BertTokenizer.from_pretrained(getattr(opt, 'local_bert_path', 'bert-base-uncased'))
        self.bert = BertModel.from_pretrained(getattr(opt, 'local_bert_path', 'bert-base-uncased'), **attention_options(opt))
        if opt.embed_size == self.bert.config.hidden_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)

    def forward(self, x, lengths):
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]
        cap_emb = self.fc(bert_emb)
        return cap_emb

class LocalVisionEncoder(VisionTransEncoder):
    """Intermediate visual vectors and the final-layer global CLS vector."""

    def __init__(self, opt):
        super().__init__(opt)
        self.captured = {}
        layer = opt.slot_readout_layer
        self.visual_encoder.encoder.layer[layer - 1].register_forward_hook(self.capture_selected)

    def capture_selected(self, module, args, out):
        self.captured['selected'] = out[0] if isinstance(out, tuple) else out

    def forward(self, images):
        self.captured.clear()
        normal = self.visual_encoder(images).last_hidden_state
        local = self.visual_encoder.layernorm(self.captured['selected'])
        return torch.cat([self.vision_proj(normal[:, :1]),
                          self.vision_proj(local[:, 1:])], dim=1)


def mask_xattn_batch(units, words, lengths):
    """Normalized inputs; score all image-text pairs, including CLS and SEP."""
    lengths = torch.as_tensor(lengths, device=words.device, dtype=torch.long)
    valid = torch.arange(words.shape[1], device=words.device)[None, :] < lengths[:, None]
    scores = torch.einsum('tlc,vmc->tvlm', words, units)
    if scores.dtype == torch.bfloat16:
        scores = scores.float()
    scores = F.leaky_relu(scores, negative_slope=0.1)
    row = scores.max(dim=-1).values.masked_fill(~valid[:, None, :], 0)
    text_side = row.sum(dim=-1) / lengths[:, None]
    visual_side = scores.masked_fill(
        ~valid[:, None, :, None], -torch.inf).max(dim=-2).values.mean(dim=-1)
    return (text_side + visual_side).transpose(0, 1)


class ContrastiveLoss(nn.Module):
    """Bidirectional hinge-based triplet loss with valid image-identity negatives."""

    def __init__(self, opt, margin=0.2, max_violation=False):
        super().__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, scores, img_ids):
        diagonal = scores.diag().view(scores.size(0), 1)
        cost_s = (self.margin + scores - diagonal.expand_as(scores)).clamp(min=0)
        cost_im = (self.margin + scores - diagonal.t().expand_as(scores)).clamp(min=0)
        mask = img_ids.unsqueeze(1) == img_ids.unsqueeze(0)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)
        if self.max_violation:
            cost_s = cost_s.max(1).values
            cost_im = cost_im.max(0).values
        return cost_s.sum() + cost_im.sum()


class SlotUnits(nn.Module):
    """Describability Gating and Competitive Semantic Aggregation."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        c = opt.embed_size
        self.keeped_patches = opt.num_slots
        self.num_image_tokens = self.keeped_patches + 1
        self.input_norm = nn.LayerNorm(c)
        self.desc_head = nn.Sequential(
            nn.LayerNorm(c), nn.Linear(c, 128), nn.GELU(), nn.Linear(128, 1))
        self.initial_slots = nn.Parameter(torch.randn(self.keeped_patches, c))
        self.query = nn.Linear(c, c, bias=False)
        self.key = nn.Linear(c, c, bias=False)
        self.key.weight = self.query.weight
        self.value = nn.Linear(c, c, bias=False)
        self.gru = nn.GRUCell(c, c)
        self.residual = nn.Sequential(
            nn.LayerNorm(c), nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c))
        # u_k = P_o m_k: no additive output bias.
        self.projection = nn.Linear(c, c, bias=False)
        self.last_desc_logits = None
        self.last_attention = None

    @staticmethod
    def gated_attention(scores, d):
        with full_precision_context():
            assignment = F.softmax(scores.float(), dim=1)
            weighted = assignment * d[:, None, :].float()
            return weighted / (weighted.sum(dim=-1, keepdim=True) + 1e-6)

    def compute_image_units(self, image):
        x = self.input_norm(image[:, 1:].float())
        logits = self.desc_head(x).squeeze(-1)
        self.last_desc_logits = logits
        participation = logits.float().sigmoid()
        keys = self.key(x)
        values = self.value(x)
        slots = self.initial_slots[None].expand(len(image), -1, -1)
        for _ in range(self.opt.slot_iters):
            scores = (self.query(slots) @ keys.transpose(1, 2)).float() / math.sqrt(x.shape[-1])
            attention = self.gated_attention(scores, participation)
            pooled = attention.to(values.dtype) @ values
            with full_precision_context():
                state = self.gru(
                    pooled.float().reshape(-1, x.shape[-1]),
                    slots.float().reshape(-1, x.shape[-1])).reshape_as(slots)
            slots = state + self.residual(state)
        units = self.projection(pooled).float()
        # Cosine normalization is applied in matching rather than unit construction.
        tokens = torch.cat([image[:, :1].float(), units], dim=1)
        self.last_attention = attention
        return tokens, attention, logits

    def score_image_units(self, units, captions, lengths):
        units = F.normalize(units.float(), dim=-1)
        words = F.normalize(captions.float(), dim=-1)
        lens = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
        length_tensor = torch.as_tensor(lengths, device=words.device, dtype=torch.long)
        scores = []
        for i in range(0, len(words), self.opt.caption_chunk_size):
            end = i + self.opt.caption_chunk_size
            scores.append(mask_xattn_batch(
                units, words[i:end, :max(lens[i:end])], length_tensor[i:end]))
        return torch.cat(scores, dim=1)


class SlotsModel(nn.Module):
    """Image-text encoders, semantic unit construction, and retrieval supervision."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        configure_precision(opt)
        self.img_enc = LocalVisionEncoder(opt)
        self.txt_enc = EncoderText_BERT(opt)
        self.cross_net = SlotUnits(opt)
        self.criterion = ContrastiveLoss(
            opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        self.Eiters = 0
        self.last_losses = {}

    def set_max_violation(self, max_violation=True):
        self.criterion.max_violation_on() if max_violation else self.criterion.max_violation_off()

    @precision_forward
    def forward_emb(self, images, captions, lengths):
        return (self.img_enc(images.cuda()).float(),
                self.txt_enc(captions.cuda(), lengths.cuda()).float(), lengths.cuda())

    @precision_forward
    def forward(self, images, captions, lengths, img_ids, warmup_alpha=1.0):
        image = self.img_enc(images)
        words = self.txt_enc(captions, lengths)
        units, _, _ = self.cross_net.compute_image_units(image)
        similarities = self.cross_net.score_image_units(units, words, lengths)
        with full_precision_context():
            loss = self.criterion(similarities.float(), img_ids) * warmup_alpha
        self.last_losses = {'align_loss': float(loss.detach())}
        return loss, None


def create_optimizer(opt, model):
    decay_factor = 0.0001
    cross_lr_rate = 1.0
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()
    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)
    params_list = [{'params': text_params_no_bert, 'lr': opt.learning_rate}, {'params': bert_params, 'lr': opt.learning_rate * 0.1}]
    params_list += [{'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1}, {'params': model.img_enc.vision_proj.parameters(), 'lr': opt.learning_rate}]
    params_list += [{'params': model.cross_net.parameters(), 'lr': opt.learning_rate * cross_lr_rate}, {'params': model.criterion.parameters(), 'lr': opt.learning_rate}]
    optimizer = torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)
    return optimizer
