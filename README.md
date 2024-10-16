## Background Inpainting
This project utilizes Grounding DINO, SAM (Segment Anything), and LAMA Inpaint to achieve seamless background replacement while preserving the integrity of the original elements.
<p align="center">
<img src="assets/image.jpeg" width="200" />
<img src="assets/mask.png" width="200" />
<img src="assets/final_image.png" width="200" />
</p>

<img src="assets/grid_sample.png"/>

## Installation

Install the dependency
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