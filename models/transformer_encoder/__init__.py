import os
import torch
import torch.nn as nn
import math

from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

from models.transformer_encoder.transformer import (
    MaskedTransformerEncoder, TransformerEncoder,
    TransformerEncoderLayer
)


class SceneTokenizeEncoder(nn.Module):
    
    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder
        self.pretrained_path = args.pretrained_path
        self._load_encoder()  # Load pretrained weights
        self._freeze_encoder(args.freeze_encoder)  # Optionally freeze encoder

    def _load_encoder(self):
        checkpoint_path = os.path.join(self.pretrained_path, "transformer_scene_encoder.pth")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # print(f"Loading pretrained weights from {checkpoint_path}")
        except FileNotFoundError:
            print("Error: Pretrained weights not found. Please download them.")
            exit(-1)

        state_dict = self.encoder.state_dict()
        pretrained_params = {name: param for name, param in checkpoint['model'].items() if name in state_dict}
        state_dict.update(pretrained_params)
        missing_keys, _ = self.encoder.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys in Scene Encoder: {missing_keys}")
        
    def _freeze_encoder(self, freeze: bool):
        if freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _split_pointcloud(self, point_cloud):
        xyz = point_cloud[..., :3].contiguous()
        features = point_cloud[..., 3:].transpose(1, 2).contiguous() if point_cloud.size(-1) > 3 else None
        return xyz, features

    def _run_encoder(self, point_clouds):
        xyz, features = self._split_pointcloud(point_clouds)

        # Tokenization and encoder processing
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.encoder[0](xyz, features)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)  
        
        enc_xyz, enc_features, enc_inds = self.encoder[1](pre_enc_features, xyz=pre_enc_xyz)
        enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long()) if enc_inds is not None else pre_enc_inds

        return enc_xyz, enc_features.permute(1, 0, 2), enc_inds

    def forward(self, batch_data):
        point_clouds = batch_data["point_clouds"]
        
        _, scene_features, _ = self._run_encoder(point_clouds)
        return {"scene_features": scene_features}  # batch x npoints x channel



def base_encoder(args):

    in_channel = (
        3   * (int(args.use_color) + int(args.use_normal)) + \
        1   * int(args.use_height) + \
        128 * int(args.use_multiview)
    )
    
    mlp_dims = [in_channel, 64, 128, args.enc_dim]

    tokenizer = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )

    encoder_layer = TransformerEncoderLayer(
        d_model=args.enc_dim,
        nhead=args.enc_nhead,
        dim_feedforward=args.enc_ffn_dim,
        dropout=args.enc_dropout,
        activation=args.enc_activation,
    )
    interim_downsampling = PointnetSAModuleVotes(
        radius=0.4,
        nsample=32,
        npoint=args.preenc_npoints // 2,
        mlp=[args.enc_dim, 256, 256, args.enc_dim],
        normalize_xyz=True,
    )
    masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
    encoder = MaskedTransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=3,
        interim_downsampling=interim_downsampling,
        masking_radius=masking_radius,
    )

    return nn.Sequential(tokenizer, encoder)


def build_scene_encoder(args, dataset_config):
    """Builds the full scene encoder using the transformer-based vision encoder."""
    # cfg = model_config(args, dataset_config)
    encoder = base_encoder(args)
    return SceneTokenizeEncoder(args, encoder)
