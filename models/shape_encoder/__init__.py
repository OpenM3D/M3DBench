import os
import torch
from torch import Tensor, nn
from typing import Dict

from third_party.pointnet2.pointnet2_modules import PointnetSAModule


class PointNetPP(nn.Module):
    """
    PointNet++ Encoder.
    Refer to the paper (https://arxiv.org/abs/1706.02413) for hyperparameters.
    """

    def __init__(self, sa_n_points: list, sa_n_samples: list, sa_radii: list, sa_mlps: list, bn=True, use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lengths of the given hyper-parameters are not compatible')

        self.encoder = nn.ModuleList()
        for i in range(n_sa):
            self.encoder.append(PointnetSAModule(
                npoint=sa_n_points[i],
                nsample=sa_n_samples[i],
                radius=sa_radii[i],
                mlp=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])
    
    @staticmethod
    def _break_up_pc(pc: Tensor) -> tuple:
        """
        Split the pointcloud into xyz positions and features tensors.
        Taken from VoteNet codebase (https://github.com/facebookresearch/votenet).

        Args:
            pc (Tensor): Pointcloud [N, 3 + C].

        Returns:
            tuple: The xyz tensor and the feature tensor.
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward pass of the PointNet++ encoder.

        Args:
            features (Tensor): B x N_objects x N_Points x (3 + C).

        Returns:
            Tensor: Encoded features.
        """
        xyz, features = self._break_up_pc(features)
        for layer in self.encoder:
            xyz, features = layer(xyz, features)

        return self.fc(features.view(features.size(0), -1))


class ShapeTokenizeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Initialize the point feature extractor
        self.point_feature_extractor = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
        )

        # Load encoder and freeze if specified
        self.pretrained_path = args.pretrained_path
        self.freeze_feature = args.freeze_encoder
        # self.dropout = nn.Dropout(args.dropout_rate)

        self._load_encoder()
        if args.freeze_encoder:
            self._freeze_encoder()

    def _load_encoder(self):
        """
        Load the pretrained weights for the encoder.
        """
        checkpoint_path = os.path.join(self.pretrained_path, "shape_encoder.pth")
        if not os.path.exists(checkpoint_path):
            print("Error: Pretrained weights not found. Please download them.")
            exit(-1)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self.point_feature_extractor.state_dict()

        # Update model state dict with pretrained weights
        pretrained_params = {name: param for name, param in checkpoint['model'].items() if name in state_dict}
        state_dict.update(pretrained_params)
        missing_keys, _ = self.point_feature_extractor.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in Shape Encoder: {missing_keys}")

    def _freeze_encoder(self):
        """
        Freeze the parameters of the encoder.
        """
        self.point_feature_extractor.eval()
        for param in self.point_feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, batch_data: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass of the ShapeTokenizeEncoder.

        Args:
            batch_data (Dict[str, Tensor]): Input batch data containing the point cloud.

        Returns:
            Tensor: The encoded features of the input point cloud.
        """
        # Get point cloud data
        obj_pcds = batch_data["shape"]

        # Extract features from point cloud
        obj_embeds = self.point_feature_extractor(obj_pcds).unsqueeze(1)


        # # Apply dropout to extracted features
        # obj_embeds = self.dropout(obj_embeds)

        # # Detach features if frozen
        # if self.freeze_feature:
        #     obj_embeds = obj_embeds.detach()

        return obj_embeds


def build_prompt_encoder(args, dataset_config):

    return ShapeTokenizeEncoder(args)
