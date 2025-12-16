import argparse
import json
import os
import torch
from multiprocessing import Process
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from tqdm import tqdm  

import sys
sys.path.append("src")
from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)


def run_dataset(model_name, dataset, device, dynamic, step):
    dataset_info = {
        "ArtiMuse-10K": ('benchmark/VR/IAA/ArtiMuse-10K/ArtiMuse-10K.json', 'benchmark/VR/IAA/ArtiMuse-10K/image', "iaa"),
        "AVA": ('benchmark/VR/IAA/AVA/AVA.json', 'benchmark/VR/IAA/AVA/image', "iaa"),
        "TAD66K": ('benchmark/VR/IAA/TAD66K/TAD66K.json', 'benchmark/VR/IAA/TAD66K/image', "iaa"),
        "FLICKR-AES": ('benchmark/VR/IAA/FLICKR-AES/FLICKR-AES.json', 'benchmark/VR/IAA/FLICKR-AES/image', "iaa"),
        "KonIQ-10K": ('benchmark/VR/IQA/KonIQ-10K/KonIQ-10K.json', 'benchmark/VR/IQA/KonIQ-10K/image', "iqa"),
        "SPAQ": ('benchmark/VR/IQA/SPAQ/SPAQ.json', 'benchmark/VR/IQA/SPAQ/image', "iqa"),
        "KADID": ('benchmark/VR/IQA/KADID/KADID.json', 'benchmark/VR/IQA/KADID/image', "iqa"),
        "PIPAL": ('benchmark/VR/IQA/PIPAL/PIPAL.json', 'benchmark/VR/IQA/PIPAL/image', "iqa"),
        "ISTA-10K": ('benchmark/VR/ISTA/ISTA-10K/ISTA-10K.json', 'benchmark/VR/ISTA/ISTA-10K/image', "ista"),
    }

    input_json, img_root, task_type = dataset_info[dataset]
    model_path = f"ckpt/{model_name}" + (f"/checkpoint-{step}" if step else "")

    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    gen_cfg = dict(max_new_tokens=512, do_sample=False)

    output_dir = "results/vr"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{dataset}_{model_name}.json"

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    progress = tqdm(
        range(len(data)),
        desc=f"{dataset}",
        ncols=100,
        position=int(os.getenv("TQDM_POS", 0)),
        leave=True
    )

    for i in progress:
        item = data[i]
        img_path = os.path.join(img_root, item["image"])
        if not os.path.exists(img_path):
            item["score"] = None
            progress.update(0)
            continue

        try:
            pixel_values = load_image(img_path).to(torch.bfloat16).to(device)
            desc = "aesthetics" if task_type == "iaa" else ("quality" if task_type == "iqa" else "structure and texture richness")
            score = model.score(device, tokenizer, pixel_values, gen_cfg, desc)
        except Exception:
            score = None

        item["score"] = score

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    devices = [f"cuda:{d}" for d in args.devices.split(",")]

    processes = []
    for i, dataset in enumerate(datasets):
        device = devices[i % len(devices)]
        os.environ["TQDM_POS"] = str(i)   
        p = Process(target=run_dataset, args=(args.model_name, dataset, device, args.dynamic, args.step))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()