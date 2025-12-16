# VQA-IAA
python src/eval/eval_vqa.py \
    --input_json benchmark/VQA/IAA/UniPercept-Bench-IAA.json \
    --model_path ckpt/unipercept \
    --output_json results/vqa/UniPercept-Bench-VQA-IAA_unipercept.json \
    --model_name unipercept

# VQA-IQA
python src/eval/eval_vqa.py \
    --input_json benchmark/VQA/IQA/UniPercept-Bench-IQA.json \
    --model_path ckpt/unipercept \
    --output_json results/vqa/UniPercept-Bench-VQA-IQA_unipercept.json \
    --model_name unipercept

# VQA-ISTA
python src/eval/eval_vqa.py \
    --input_json benchmark/VQA/ISTA/UniPercept-Bench-ISTA.json \
    --model_path ckpt/unipercept \
    --output_json results/vqa/UniPercept-Bench-VQA-ISTA_unipercept.json \
    --model_name unipercept
