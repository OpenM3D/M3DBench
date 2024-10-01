import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class ImageTokenizeEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Attributes
        self.is_loaded = False
        self.mm_hidden_size = 1024
        self.n_embd = 4096
        self.image_encoder = args.image_encoder
        self.llava_projector = os.path.join(args.pretrained_path, "mm_projector.bin")
        self.select_layer = -2
        self.select_feature = 'cls'

        # Initialize Projector
        self.projector = nn.Sequential(
            nn.Linear(self.mm_hidden_size, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd)
        )

        # Load encoder and projector
        self._load_encoder()
        self._load_projector()
        self._freeze()

    def _freeze(self, mode: bool = True):
        """
        Train method that controls freezing the model parameters.
        """
        super().train(mode)

        if mode:  
            self._freeze_model(self.vision_tower)
            self._freeze_model(self.projector)

    def _freeze_model(self, module: nn.Module):
        """
        Helper function to freeze the parameters of a given module.
        """
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _load_encoder(self):
        """
        Load the vision tower and image processor from pretrained model.
        """
        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_encoder)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.image_encoder)
        self.is_loaded = True

    def _load_projector(self):
        """
        Load the weights of the projector from a pretrained checkpoint.
        """
        model_state_dict = self.projector.state_dict()
        state_dict = torch.load(self.llava_projector)

        # Load state dict while removing prefix "model.mm_projector."
        for key in state_dict.keys():
            new_key = key.replace("model.mm_projector.", "")
            if new_key in model_state_dict:
                model_state_dict[new_key] = state_dict[key]

        # Load updated state dict into projector
        missing_keys, _ = self.projector.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys in Image Encoder: {missing_keys}")

    def feature_select(self, image_forward_outs):
        """
        Select features from the image encoder output based on configuration.
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # Exclude CLS token
        elif self.select_feature == 'cls':
            image_features = image_features[:, 0]  # Only use CLS token
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return image_features

    @torch.no_grad()
    def forward(self, batch_data):
        """
        Forward pass for processing input images and adding projected features to outputs.
        """
        images = batch_data['image']
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            image_features = torch.cat(image_features, dim=0)  # Concatenate features from the batch list
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # Project image features and add them to the output
        image_features = self.projector(image_features).unsqueeze(1)
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            raise ValueError("Encoder is not loaded. Configuration is unavailable.")

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_prompt_encoder(args):
    """
    Build the prompt encoder model based on provided arguments.
    """
    return ImageTokenizeEncoder(args)
