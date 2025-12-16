import argparse
import json
import os
import math
from PIL import ImageFile, Image
import torch
from transformers import AutoTokenizer
import sys
sys.path.append("src")
from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def infer(model, tokenizer, gen_cfg, image_path, question):
    prompt = (
        "Given the image and the following multiple-choice question, choose the best answer. "
        "Only reply with the letter (A, B, C, or D).\n\n"
        f"Question:\n\n{question}"
    )

    pixel_values = load_image(image_path).to(torch.bfloat16).to(DEVICE)

    response = model.chat(
        DEVICE, tokenizer, pixel_values,
        prompt, gen_cfg,
        history=None,
        return_history=False
    )

    return response.strip()

def evaluate_and_save(input_data, model, tokenizer, gen_cfg, model_name, output_json):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    total = len(input_data)
    correct = 0

    for idx, item in enumerate(input_data):
        image_path = item["image"]
        question = item["question"]
        gt = item["answer"].strip().upper()

        try:
            pred_raw = infer(model, tokenizer, gen_cfg, image_path, question)
        except Exception as e:
            print(f"[!] Failed on {image_path}: {e}")
            pred_raw = "?"

        pred = pred_raw.strip().upper()[:1]  
        item[f"answer_{model_name}"] = pred
        correct += int(pred == gt)

        print(f"[{model_name}] Q{idx + 1:04d}: pred={pred}, gt={gt}  {'✔' if pred == gt else '✘'}")

    acc = correct / total
    with open(output_json, "w", encoding="utf-8") as fw:
        json.dump(input_data, fw, ensure_ascii=False, indent=2)

    print(f"\n📌 Saved to: {output_json}")
    print(f"🎯 Final Accuracy = {correct}/{total} = {acc:.2%}")

def main():
    parser = argparse.ArgumentParser(description="UniPercept VQA Evaluation Script")
    parser.add_argument("--input_json", required=True, help="Input benchmark JSON file")
    parser.add_argument("--model_path", required=True, help="Model checkpoint dir or HF model name")
    parser.add_argument("--model_name", required=True, help="Name used as key in JSON results")
    parser.add_argument("--output_json", required=True, help="Where to save JSON with predictions")
    args = parser.parse_args()

    print("🔍 Loading dataset...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    model, tokenizer, gen_cfg = load_model(args.model_path)
    evaluate_and_save(data, model, tokenizer, gen_cfg, args.model_name, args.output_json)


if __name__ == "__main__":
    main()