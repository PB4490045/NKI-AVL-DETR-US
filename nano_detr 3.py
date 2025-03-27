import torch.nn as nn

from torch.profiler import record_function

import torch
import timm

class NanoDETR(nn.Module):
    def __init__(self, config, hidden_dim=None, num_encoder_layers=1, num_decoder_layers=1, num_preds=1):
        super().__init__()
        # constants
        self.patch_dim = config.PATCH_DIM
        self.mean = torch.tensor([1]).cuda()
        self.std = torch.tensor([0.147]).cuda()
        # backbone to generate feature maps from input frame
        self.backbone = timm.create_model(
            # 'efficientvit_b0.r224_in1k',
            'convnextv2_atto.fcmae_ft_in1k',
            pretrained=config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool='',
        )
        # backbone hidden dimension
        self.hidden_dim_backbone = self.backbone.feature_info[-1]['num_chs']
        # dimension in detection transformer
        if hidden_dim is None:
            self.hidden_dim = self.backbone.feature_info[-1]['num_chs']
        else:
            self.hidden_dim = hidden_dim
        # compression layer
        self.proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim_backbone, self.hidden_dim, kernel_size=1),
            )
        # Generic transformer for query with feature map interaction
        self.transformer = nn.Transformer(
                d_model=self.hidden_dim,
                dim_feedforward=self.hidden_dim*4,
                nhead=self.hidden_dim//16,
                dropout=0.00,
                activation='gelu',
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                batch_first=True,
            )
        # Classification layers
        self.linear_class = nn.Sequential(
                nn.Linear(self.hidden_dim, config.N_CLASSES),
            )
        self.linear_bbox = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 4),
            )
        # output positional encodings
        self.query_pos = nn.Parameter(torch.randn(1, num_preds, self.hidden_dim), requires_grad=True)
        # learnable spatial positional encodings
        self.pe_h = nn.Parameter(torch.randn(self.hidden_dim // 2, config.PATCH_DIM), requires_grad=True)
        self.pe_w = nn.Parameter(torch.randn(self.hidden_dim // 2, config.PATCH_DIM), requires_grad=True)

    def forward(self, inputs, debug=False): 
        # Normalize Input
        inputs = inputs.float() / 255.0
        # inputs = (inputs - self.mean) / self.std
        inputs = inputs.swapaxes(3, 1)
        # Acquire Batch Size
        B = inputs.shape[0]
        # Backbone
        x_backbone = self.backbone(inputs)
        # Projection to hidden dim
        x_proj = self.proj(x_backbone)
        # Positional Encoding
        pos = torch.cat([
            self.pe_h.unsqueeze(1).repeat(1, self.patch_dim, 1),
            self.pe_w.unsqueeze(2).repeat(1, 1, self.patch_dim),
        ], dim=0).unsqueeze(0)
        # construct positional encodings
        x_proj = (x_proj + pos).flatten(2).permute(2, 0, 1)

        # propagate through the transformer
        x = self.transformer(
                x_proj.swapaxes(1,0),
                self.query_pos.repeat(B,1,1),
            )
        # objectness prediction
        linear_cls = self.linear_class(x).squeeze(1)
        # coordinate prediction
        bb_xyhw = self.linear_bbox(x).squeeze(1).sigmoid()
        # convert xyhw → xyxy
        x_c, y_c, w, h = torch.chunk(bb_xyhw, chunks=4, dim=1)
        bb_xyxy = torch.concat([x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2], axis=1)
        # result
        if debug:
            return { 'obj': linear_cls, 'bb_xyhw': bb_xyhw, 'bb_xyxy': bb_xyxy, 'x_backbone': x_backbone, 'x_proj': x_proj }
        else:
            return { 'obj': linear_cls, 'bb_xyhw': bb_xyhw, 'bb_xyxy': bb_xyxy }
