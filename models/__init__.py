import torch
import importlib
from torch import nn

class M3DAssistant(nn.Module):
    """
    Baseline model combining a scene encoder, multi-modal prompt encoder, and a decoder.
    This model can be used in both training and evaluation modes.
    """
    def __init__(self, args, dataset_config, train_dataset):
        super(M3DAssistant, self).__init__()

        # Scene encoder
        scene_encoder_module = importlib.import_module(f'models.{args.vision_encoder}_encoder')
        self.scene_encoder = scene_encoder_module.build_scene_encoder(args, dataset_config)

        # # Multimodal prompt encoders
        image_encoder_module = importlib.import_module('models.image_encoder')
        self.image_encoder = image_encoder_module.build_prompt_encoder(args)
        shape_encoder_module = importlib.import_module('models.shape_encoder')
        self.shape_encoder = shape_encoder_module.build_prompt_encoder(args, dataset_config)

        # Decoder
        backbone_module = importlib.import_module(f'models.{args.task_decoder}_decoder')
        self.backbone = backbone_module.build_backbone(args, train_dataset)

    def forward(self, batch_data: dict, is_eval: bool = False) -> dict:
        """
        Forward pass through the model, which includes:
        - Scene encoding
        - Multimodal prompt encoding (image and shape prompts)
        - Decoding

        Args:
            batch_data (dict): Batch input data with labels.
            is_eval (bool): Evaluation mode flag.

        Returns:
            dict: Output dictionary with results and loss.
        """

        outputs = self.scene_encoder(batch_data)

        outputs['image_features'] = self.image_encoder(batch_data)
        outputs['shape_features'] = self.shape_encoder(batch_data)

        # Initialize loss on the same device as the input data
        if 'loss' not in outputs:
            outputs['loss'] = torch.zeros(1, device='cuda')

        final_outputs = self.backbone(outputs, batch_data, is_eval=is_eval)

        return final_outputs


def build_assistant(args, dataset_config, train_dataset):
    """
    Args:
        args: Arguments specifying model components and configuration.
        dataset_config: Dataset configuration needed for building the model.
        train_dataset: Training dataset to be passed to the model.
    
    Returns:
        baseline model.
    """
    model = M3DAssistant(args, dataset_config, train_dataset)
    return model
