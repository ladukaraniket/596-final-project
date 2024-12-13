import os
import sys
sys.path.append('content/ai-toolkit')
from collections import OrderedDict
# from PIL import Image
from diffusers import AutoPipelineForText2Image
import torch

os.environ['HF_TOKEN'] = "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"
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)

pipeline.to("cuda")

image = pipeline('a woman holding a coffee cup, in a beanie, sitting at a cafe').images[0]
image.save('output.png')
