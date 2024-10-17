## Background Inpainting
This project utilizes Stable Diffusion XL, ControlNet, Grounding DINO, SAM (Segment Anything), and LAMA Inpaint to achieve seamless background replacement while preserving the integrity of the original elements.

<div align="center">
<img src="assets/image.jpeg" width="20%" />
<img src="assets/mask.png" width="20%" />
<img src="assets/final_image.png" width="20%" />
</div>

<img src="assets/grid_sample.png"/>

Specifically, GroundingDINO is used to identify the object, which are then segmented with SAM. The background is inpainted using ControlNet and Stable Diffusion XL, and the foreground is removed via LAMA inpainting. At last, the identified object are copied and pasted into the newly generated background.

## Installation

Before installing diffusers, make sure that `PyTorch` and `Accelerate` are already installed.
```bash
pip install diffusers["torch"] transformers
```

Install the SAM library and LAMA inpaint dependencies.
```bash
pip install -r requirements.txt
```

Install GroundingDINO from the git repository.
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

You can download the pre-trained model weights from the following link.

- `GroundingDINO`: [Swin-T GroundingDINO model](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- `SAM`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `LAMA Inpaint`: [Big LAMA](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing)


## Usage
Run the following command, and all generated images will be saved in the `output_dir`.
```bash
python inference.py \
    --input_img ./assets/image.jpeg \
    --output_dir ./output \
    --extract_prompt "person" \
    --background_prompt "frozen landscape with a glacier in the background, snow and wind swirling through the air" \
    --negative_prompt "low quality, bad quality, sketches" \
    --ground_ckpt ./pretrained_models/groundingdino_swint_ogc.pth \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_ckpt ./pretrained_models/big-lama
```

## Acknowledgement
This code is based on [Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/controlnet#controlnet-with-stable-diffusion-xl), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/segment-anything), [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything), .