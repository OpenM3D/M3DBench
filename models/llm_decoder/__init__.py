import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict
from transformers import AutoModelForCausalLM

from models.llm_decoder.generation_utils import generation
from models.llm_decoder.projectors import SceneProj, ClickProj, RegionProj, ShapeProj, ImageProj



class llm_backbone(nn.Module):
    def __init__(self, args, train_dataset):
        super(llm_backbone, self).__init__()

        self.max_des_len = args.max_des_len

        # Initialize tokenizer for batch decoding
        self.tokenizer = train_dataset.tokenizer
        self.dtype = torch.float16

        # Initialize response generation cores
        self.transformer = AutoModelForCausalLM.from_pretrained(
            args.base_llm_path,
            torch_dtype=self.dtype
        )

        self.n_embd = self.transformer.config.hidden_size
        self.nvocabs = self.transformer.config.vocab_size

        # Load projection layers
        self.scene_proj = SceneProj.build(args.enc_dim, self.n_embd)
        self.click_proj = ClickProj.build(self.n_embd)
        self.region_proj = RegionProj.build(self.n_embd)
        self.shape_proj = ShapeProj.build(self.n_embd)
        self.image_proj = ImageProj.build(self.n_embd)

        self.response_config = {
            'max_length': self.max_des_len,
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 10 if args.use_beam_search else None,
        }

        self._freeze_decoder(args.freeze_decoder)  # Optionally freeze encoder


    def _freeze_decoder(self, freeze: bool):
        if freeze:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False

    def _prepare_representations(self, output: dict, inputs: dict) -> dict:
        # Using the projection methods from respective classes
        output['scene_features'] = self.scene_proj(output['scene_features'])
        output['shape_features'] = self.shape_proj(output['shape_features'])
        output['click_features'] = self.click_proj(inputs['click'])
        output['image_features'] = self.image_proj(output['image_features'])
        output['region_features'] = self.region_proj(inputs['region'])
        return output

    def _forward_training(self, output: Dict, inputs: Dict) -> Dict:
        
        scene_features = output['scene_features']
        image_features = output['image_features']
        shape_features = output['shape_features']
        click_features = output['click_features']
        region_features = output['region_features']

        total_ids = inputs['total_ids']  # batch x ntokens
        attention_mask = inputs['total_mask']  # batch x ntokens
        gradient_mask = inputs['answer_mask']  # batch x ntokens

        # Prepare multimodal masks
        click_mask = inputs['click_mask'].repeat(1, click_features.size(1))
        image_mask = inputs['image_mask'].repeat(1, image_features.size(1))
        shape_mask = inputs['shape_mask'].repeat(1, shape_features.size(1))
        region_mask = inputs['region_mask'].repeat(1, region_features.size(1))

        # Convert features and masks to the correct dtype
        scene_features, image_features, shape_features, click_features, region_features = [
            feat.to(self.dtype) for feat in [scene_features, image_features, shape_features, click_features, region_features]
        ]
        attention_mask, gradient_mask, click_mask, image_mask, shape_mask, region_mask = [
            mask.to(self.dtype) for mask in [attention_mask, gradient_mask, click_mask, image_mask, shape_mask, region_mask]
        ]

        # Concatenate all multimodal features
        prefix_tokens = torch.cat((scene_features, image_features, shape_features, click_features, region_features), dim=1)
        prefix_mask = torch.cat((torch.ones_like(scene_features[..., 0]), image_mask, shape_mask, click_mask, region_mask), dim=1)

        # Get word embeddings and concatenate with multimodal features
        embedding_layer = self.transformer.get_input_embeddings()
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(total_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)

        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Compute loss
        output['loss'] = 5 * self.loss_response(
            logits=outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target=total_ids.long(),
            mask=gradient_mask
        )

        return output
    
    
    def loss_response(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss_per_word = nnf.cross_entropy(
            logits.reshape(-1, self.nvocabs),
            target.reshape(-1), 
            reduction='none',
        )
        loss_per_word = loss_per_word.reshape(target.shape)
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)
        return final_loss
    
    
    def _forward_evaluation(self, output: Dict, inputs: Dict) -> Dict:
        
        query_ids = inputs['question_ids']
        query_mask = inputs['question_mask']

        # Prepare prefix tokens
        scene_features = output['scene_features'].to(self.dtype)
        image_features = output['image_features'].to(self.dtype)
        shape_features = output['shape_features'].to(self.dtype)
        click_features = output['click_features'].to(self.dtype)
        region_features = output['region_features'].to(self.dtype)

        prefix_tokens = torch.cat((scene_features, image_features, shape_features, click_features, region_features), dim=1)

        batch_size = prefix_tokens.shape[0]
        embedding_layer = self.transformer.get_input_embeddings()

        output_ids = []
        for batch_id in range(batch_size):
            query = query_ids[batch_id]
            mask = query_mask[batch_id]

            output = generation(
                self.transformer,
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[[batch_id]],
                        embedding_layer(query[mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                **self.response_config
            )
            output_ids.append(output['output_ids'])

        output_ids = torch.cat(output_ids, dim=0)
        output['output_ids'] = output_ids

        return output
    
    def forward(self, output: dict, inputs: dict, is_eval: bool = False) -> dict:
        output = self._prepare_representations(output, inputs)
        if is_eval:
            return self._forward_evaluation(output, inputs)
        else:
            return self._forward_training(output, inputs)
    

def build_backbone(args, train_dataset):
    model = llm_backbone(args, train_dataset)
    return model
