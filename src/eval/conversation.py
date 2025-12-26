import argparse
import os
import sys
import torch
from PIL import Image, ImageFile
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append("src")
from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

DEVICE = "cuda:0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


TRANSFORM = build_transform(448)


def load_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


def load_model(model_path: str):
    print(f"🔹 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).to(DEVICE).eval()

    gen_cfg = dict(
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    return model, tokenizer, gen_cfg


def main():
    parser = argparse.ArgumentParser(description="Chat with UniPercept")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    model, tokenizer, gen_cfg = load_model(args.model_path)

    pixel_values = None
    curr_prompt = args.prompt
    if args.image:
        pixel_values = load_image(args.image).to(torch.bfloat16).to(DEVICE)
        curr_prompt = f"<image>\n{args.prompt}"

    with torch.no_grad():
        response = model.chat(
            DEVICE, 
            tokenizer, 
            pixel_values, 
            curr_prompt, 
            gen_cfg, 
            history=None, 
            return_history=False
        )

    print(f"\n[User]: {args.prompt}")
    print(f"[UniPercept]: {response.strip()}\n")


if __name__ == "__main__":
    main()