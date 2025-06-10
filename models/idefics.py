# models/clip.py
from typing import List
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import Normalize
from transformers import AutoProcessor, IdeficsForVisionText2Text, BitsAndBytesConfig 
from open_flamingo.eval.eval_model import BaseEvalModel
from typing import Optional
import torch.nn.functional as F
from transformers.utils import quantization_config

class EvalModel(BaseEvalModel):
    """
    CLIP evaluation model for adversarial attack.
    Uses separate encoders for text and image.
    """
    def __init__(self, model_args):
        
        assert "model_name_or_path" in model_args and "device" in model_args, \
            "CLIP requires model_name_or_path and device."
        self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        self.device = model_args["device"] if isinstance(model_args["device"], str) else f"cuda:{model_args['device']}"
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b",use_fast=False,  image_processor_kwargs={"size": {"height": 224, "width": 224}})
        self.model = IdeficsForVisionText2Text.from_pretrained(
            "HuggingFaceM4/idefics-9b", device_map=self.device)#.to(self.device)  #quantization_config= self.quantization_config,
        self.tokenizer = self.processor.tokenizer
        self.model.eval()
        # Set flag for targeted attack; if True, loss will be computed as negative similarity.
        self.target_attack = True

    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
            all_batch_images = []
            for example in batch:  # example is a list of PIL images for one data point
                example_tensors = []
                for img in example:
                    proc_out = self.processor.image_processor(img, return_tensors="pt")
                    if isinstance(proc_out, dict):
                        img_tensor = proc_out["pixel_values"]
                    else:
                        img_tensor = proc_out
                    example_tensors.append(img_tensor)
                # Concatenate images for this example along the 0-th dimension: shape = [num_images, 3, H, W]
                example_tensor = torch.cat(example_tensors, dim=0)
                # Add a batch dimension: now shape = [1, num_images, 3, H, W]
                all_batch_images.append(example_tensor.unsqueeze(0))
            # Concatenate all examples along the batch dimension.
            final_images= torch.cat(all_batch_images, dim=0)
            return final_images #[1,batch_size,3,224,224]

    def _prepare_images_no_normalize(self, batch: List[List[Image.Image]]) -> torch.Tensor:
            all_batch_images = []
            for example in batch:  # example is a list of PIL images for one data point
                example_tensors = []
                for img in example:
                    proc_out = self.processor.image_processor(img, do_normalize=False, return_tensors="pt")
                    if isinstance(proc_out, dict):
                        img_tensor = proc_out["pixel_values"]
                    else:
                        img_tensor = proc_out
                    example_tensors.append(img_tensor)
        
                example_tensor = torch.cat(example_tensors, dim=0)
                
                all_batch_images.append(example_tensor.unsqueeze(0))
      
            final_images= torch.cat(all_batch_images, dim=0)
  
            return final_images
    
    def get_outputs(

        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        encodings = self.processor.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values=self._prepare_images(batch_images).to(self.device),
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_outputs_attack(self,
                       attack: torch.Tensor,
                       batch_text: List[str],
                       batch_images: List[List[Image.Image]],
                       max_generation_length: int,
                       num_beams: int,
                       length_penalty: float) -> List[str]:
            # Tokenize text prompt
            encodings = self.tokenizer(
                batch_text,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=2000,
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # Prepare images using IDEFICS's image processor
            pixel_values = self._prepare_images_no_normalize(batch_images).to(self.device)
            # Add adversarial noise
            pixel_values = pixel_values + attack
            
            IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
            IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]
            # Apply normalization (choose the appropriate normalization for IDEFICS)
            normalizer = Normalize(mean=IDEFICS_STANDARD_MEAN,
                            std=IDEFICS_STANDARD_STD)
            pixel_values = normalizer(pixel_values)

            
            with torch.inference_mode():
                # Now call processor with *one* positional arg
                inputs = self.processor(
                    batch_images,                         
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    add_end_of_utterance_token=False
                ).to(self.device)

                # (B, N, 3, H, W)
                image_attention_mask = inputs.image_attention_mask  # (B, N, num_patches+1)
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    image_attention_mask= image_attention_mask
                )
            
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


    def get_vqa_prompt(self, question, answer=None) -> str:
        #return f"Question: {question} Answer: {answer if answer is not None else ''}"
        return  f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:{answer if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return ""

    def get_classification_prompt(self, class_str=None) -> str:
        return ""
