from typing import NamedTuple
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionInpaintPipeline, \
    EulerAncestralDiscreteScheduler


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ZoomModels(NamedTuple, metaclass=Singleton):



    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "parlance/dreamlike-diffusion-1.0-inpainting",
            torch_dtype=torch.float16,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        def no_check(images, **kwargs):
            return images, False

        pipe.safety_checker = no_check
        pipe.enable_attention_slicing()
        self.pipe = pipe

    def model_fn(self):
        return lambda prompt, current_image, mask_image: pipe(prompt=prompt,
             negative_prompt=negative_prompt,
             image=current_image,
             guidance_scale=guidance_scale,
             height=size,
             width=size,
             mask_image=mask_image,
             num_inference_steps=num_inference_steps)
