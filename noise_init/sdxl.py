from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from IPython.display import display

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load the pipeline with half precision for speed if using GPU.
pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
target="unknown<|endofchunk|>"
result = pipe(prompt=target[:-14], num_inference_steps=50, guidance_scale=7.5).images[0]