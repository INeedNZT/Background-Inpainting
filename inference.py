import sys
import argparse
import numpy as np
import torch

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import make_image_grid
from PIL import Image
from pathlib import Path

from grounding_dino import load_model, get_grounding_output, np2tensor

from lama_inpaint import inpaint_img_with_lama
from utils import (
    load_img_to_array, 
    save_array_to_img,
    dilate_mask, 
    copy_paste, 
    ext_mask_from_img, 
    get_canny, 
    save_img
)

# segment anything
from segment_anything import (
    SamPredictor,
    build_sam
)


GROUND_CONFIG = "./config/GroundingDINO_SwinT_OGC.py"
LAMA_CONFIG = "./lama/configs/prediction/default.yaml"
DILATE_KERNEL_SIZE = 15
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 200


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input image",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results",
    )
    parser.add_argument(
        "--extract_prompt", type=str, required=True,
        help="Text prompt for the GroundingDINO model to find objects in the image",
    )
    parser.add_argument(
        "--background_prompt", type=str, required=True,
        help="Text prompt for generating the background using Stable Diffusion XL",
    )
    parser.add_argument(
        "--negative_prompt", type=str, required=True,
        help="Text prompt for specifying elements to avoid in the generated image using Stable Diffusion XL",
    )
    parser.add_argument(
        "--ground_ckpt", type=str, required=True,
        help="The path to the GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the LAMA inpaint checkpoint",
    )
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])


    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_image = load_img_to_array(args.input_img)
    H, W, _ = raw_image.shape

    model = load_model(GROUND_CONFIG, args.ground_ckpt, device=device)

    # operate on original image
    image_tensor = np2tensor(raw_image)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image_tensor, args.extract_prompt, BOX_THRESHOLD, TEXT_THRESHOLD, device=device
    )

    # scale the box filter as its been resized
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=args.sam_ckpt).to(device))
    predictor.set_image(raw_image)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    masks = masks.to(torch.uint8)
    masks = masks.cpu().numpy()
    mask = masks[0].squeeze()

    white_image = ext_mask_from_img(mask, raw_image)
    save_img(white_image, out_dir / "white_image.png")

    canny_image = get_canny(white_image, LOW_THRESHOLD, HIGH_THRESHOLD)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()


    sample_image = pipe(
        args.background_prompt,
        negative_prompt=args.negative_prompt,
        image=canny_image,
        controlnet_conditioning_scale=0.5,
    ).images[0]
    sample_image = sample_image.resize((W, H))
    save_img(sample_image,  out_dir / "sample_image.png")


    # crop mask to image size
    start_y = (mask.shape[0] - H) // 2
    start_x = (mask.shape[1] - W) // 2
    mask = mask[start_y:start_y + H, start_x:start_x + W]
    white_image = white_image[start_y:start_y + H, start_x:start_x + W]

    mask_dilate = dilate_mask(mask, DILATE_KERNEL_SIZE)
    save_img(mask*255,  out_dir / "mask.png")


    img_inpainted = inpaint_img_with_lama(
        np.array(sample_image), mask_dilate, LAMA_CONFIG, args.lama_ckpt, device=device)
    save_img(img_inpainted, out_dir / "bg_rm_image.png")

    final_image = copy_paste(mask, white_image, img_inpainted)
    save_img(final_image, out_dir / "final_image.png")

    grid_image = make_image_grid([Image.fromarray(raw_image), Image.fromarray(white_image), canny_image, sample_image, Image.fromarray(img_inpainted), Image.fromarray(final_image)], rows=1, cols=6)
    save_img(grid_image, out_dir / "grid_sample.png")

