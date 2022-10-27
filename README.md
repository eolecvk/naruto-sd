# Experiments with Stable Diffusion

## About

This repository is a fork of [Justin Pinkney's stable diffusion repository](https://github.com/justinpinkney/stable-diffusion).

It presents the use case of fine-tuning a text2image stable diffusion model with a [BLIP captioned naruto face dataset](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions).  
In this case, we generate Naruto style images from a text prompt:

<img src="./assets/outputv2_grid.png" alt="drawing" width="600"/>

> "Bill Gates with a hoodie", "John Oliver with Naruto style", "Hello Kitty with Naruto style", "Lebron James with a hat", "Mickael Jackson as a ninja", "Banksy Street art of ninja"

For a step by step guide see the [Lambda Labs examples repo](https://github.com/LambdaLabsML/examples).


## Usage


```bash
!pip install diffusers==0.3.0
!pip install transformers scipy ftfy
```

```python
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-naruto-diffusers", torch_dtype=torch.float16)  
pipe = pipe.to("cuda")

prompt = "Yoda"
scale = 7.5
n_samples = 2

# Sometimes the nsfw checker is confused by the Naruto images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

for idx, im in enumerate(images):
  im.save(f"{idx:06}.png")
```
