from diffusers import AutoPipelineForText2Image
import torch



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")
if device == "cuda":
	pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
else:
	pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
pipe = pipe.to(device)

prompt = "portrait of a young women, blue eyes, cinematic"
negative_prompt = "low quality, bad quality"

image = pipe(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale=0.75, height=256, width=256).images[0]
image.save("portrait.png")
