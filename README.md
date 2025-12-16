
<div align="center">

# UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

<a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-UniPercept-red?logo=arxiv" height="25" />
</a>
<a href="https://unipercept.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-UniPercept.github.io-blue" height="25" />
</a>
<a href="https://huggingface.co/collections/unipercept/models" target="_blank">
    <img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-UniPercept-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/unipercept/unipercept-bench" target="_blank">
    <img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-UniPercept--Bench-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

<div style="font-family: charter; text-align: center;">
<a href="#" target="_blank">Shuo Cao</a><sup>1,2*</sup>,
<a href="#" target="_blank">Jiayang Li</a><sup>3*</sup>,
<a href="#" target="_blank">Xiaohui Li</a><sup>2,4</sup>,
<a href="#" target="_blank">Yuandong Pu</a><sup>2,4</sup>,
<a href="#" target="_blank">Kaiwen Zhu</a><sup>2,4</sup>,
<a href="#" target="_blank">Yuanting Gao</a><sup>5</sup>,
<a href="#" target="_blank">Siqi Luo</a><sup>2,4</sup>,
<a href="#" target="_blank">Yi Xin</a><sup>2,6</sup>,
<a href="#" target="_blank">Qi Qin</a><sup>2</sup>,
<a href="#" target="_blank">Yu Zhou</a><sup>7</sup>,
<a href="#" target="_blank">Xiangyu Chen</a><sup>8</sup>,
<a href="#" target="_blank">Wenlong Zhang</a><sup>2</sup>,
<a href="#" target="_blank">Bin Fu</a><sup>2</sup>,
<a href="#" target="_blank">Yu Qiao</a><sup>2</sup>,
<a href="#" target="_blank">Yihao Liu</a><sup>2&#8224;</sup>
<br>
<div style="font-size: 0.9em; margin-top: 0.5em;">
<span><sup>1</sup> University of Science and Technology of China</span> &emsp;
<span><sup>2</sup> Shanghai AI Laboratory</span> &emsp;
<span><sup>3</sup> Peking University</span> &emsp;
<span><sup>4</sup> Shanghai Jiao Tong University</span> <br>
<span><sup>5</sup> Tsinghua University</span> &emsp;
<span><sup>6</sup> Nanjing University</span> &emsp;
<span><sup>7</sup> Sun Yat-sen University</span> &emsp;
<span><sup>8</sup> Tele-AI</span>
</div>
<div style="font-size: 0.8em; margin-top: 0.5em; font-style: italic;">
<span>* Equal contribution</span> &emsp;
<span>&#8224; Corresponding author</span>
</div>
</div>

</div>

<p align="center">
    <img src="asserts/img/teaser_v4.jpg" alt="Dataset Distribution" width="1000" height="auto">
</p>

## 🚀 Release
- **[Today]** 🔥 **Grand Release!** We are proud to release the full **UniPercept** suite:
    - **UniPercept-Bench**: A comprehensive perceptual-level MLLM benchmark spanning IAA, IQA, and ISTA domains, supporting both VR and VQA tasks.
    - **UniPercept**: A powerful baseline MLLM for perceptual image understanding, empowered by Domain-Adaptive Pre-Training and Task-Aligned RL.
    - **Technical Report**: The full paper is now available on arXiv.
    - **Project Page**: Visualizations and interactive demos are now live.

## 📖 Contents
- [Abstract](#-abstract)
- [UniPercept-Bench](#-unipercept-bench)
- [UniPercept](#-unipercept)
  - [Model Card](#model-card)
  - [Performance](#performance)
  - [Evaluation](#evaluation)
  - [Applications](#applications)
- [Citation](#-citation)

## 🌟 Abstract

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding tasks such as visual grounding, segmentation, and captioning. However, their ability to perceive **perceptual-level** image features remains limited. In this work, we present **UniPercept-Bench**, a unified framework for *perceptual-level image understanding* across three key domains: **Aesthetics**, **Quality**, and **Structure and Texture**. We establish a hierarchical definition system and construct large-scale datasets to evaluate perceptual-level image understanding. Based on this foundation, we develop a strong baseline **UniPercept** trained via Domain-Adaptive Pre-Training and Task-Aligned RL, enabling robust generalization across both **Visual Rating (VR)** and **Visual Question Answering (VQA)** tasks. UniPercept outperforms existing MLLMs on perceptual-level image understanding and can serve as a **plug-and-play reward model** for text-to-image generation. This work defines Perceptual-Level Image Understanding in the era of MLLMs and, through the introduction of a comprehensive benchmark together with a strong baseline, provides a solid foundation for advancing perceptual-level multimodal image understanding.


<!-- <p align="center">
    <img src="figs/comparison.png" alt="Semantic vs Perceptual Understanding" width="800" height="auto">
</p> -->

## 📊 UniPercept-Bench

We introduce **UniPercept-Bench**, a systematically designed benchmark for evaluating perceptual-level image understanding.

* **Hierarchical Taxonomy**: Organized into **Domain $\rightarrow$ Category $\rightarrow$ Criterion** layers.
* **Coverage**: 3 Domains (IAA, IQA, ISTA), 17 Categories, and 43 Criterions.
* **Tasks**:
    * **Visual Rating (VR)**: Quantitative scoring of perceptual attributes.
    * **Visual Question Answering (VQA)**: Fine-grained reasoning about visual properties.

**Download**: 🤗 [UniPercept-Bench](https://huggingface.co/datasets/unipercept/unipercept-bench)
<p align="center">
    <img src="asserts/img/unipercetp-bench.png" alt="Dataset Distribution" width="1000" height="auto">
</p>

## 🤖 UniPercept

**UniPercept** is a strong baseline MLLM trained via Domain-Adaptive Pre-Training and Task-Aligned RL to handle both **Visual Rating (VR)** (continuous scoring) and **Visual Question Answering (VQA)** (reasoning).

### Model Card

| Model | Base-MLLM | Download |
| :--- | :--- | :--- |
| **UniPercept** | `InternVL3-8B` | [🤗 Hugging Face](https://huggingface.co/unipercept/UniPercept-8B) |

## ⚙️ Setup

Clone this repository:

```
git clone https://github.com/thunderbolt215/UniPercept.git
```
Create a conda virtual environment and activate it: (please ensure that `Python>=3.9`).

```
conda create -n unipercept python=3.10
conda activate unipercept
```

Install dependencies using `requirements.txt`:
```
pip install -r requirements.txt
```
We recommend to use FlashAttention for acceleration:
```
pip install flash-attn --no-build-isolation
```

### Evaluation

We provide a unified evaluation suite for UniPercept-Bench on **Visual Rating (VR)** and **Visual Question Answering (VQA)**.

```
# Evaluate Visual Rating
bash src/eval/eval_vr.sh 

# Evaluate Visual Question Answering
bash src/eval/eval_vqa.sh 
```

### Performance

UniPercept consistently outperforms proprietary models (e.g., GPT-4o, Gemini-2.5-Pro), leading open-source models (InternVL3, Qwen3-VL) and across all three perceptual domains (IAA, IQA, ISTA) and all tasks (VR, VQA).

<p align="center">
    <img src="asserts/img/vr.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/vqa-iaa.png" alt="Performance on UniPercept-Bench-VQA (IAA)" width="1000" height="auto">
    <img src="asserts/img/vqa-iqa.png" alt="Performance on UniPercept-Bench-VQA (IQA)" width="1000" height="auto">
    <img src="asserts/img/vqa-ista.png" alt="Performance on UniPercept-Bench-VQA (ISTA)" width="1000" height="auto">
</p>

### Applications

**UniPercept As Reward**: UniPercept can be used as a powerful reward model for post-training Text-to-Image (T2I) models. By integrating UniPercept rewards into the training of **FLUX.1-dev**, we observe significant improvements in aesthetic quality, structural richness, and prompt adherence.

<p align="center">
    <img src="asserts/img/reward.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
</p>

**UniPercept As Metrics**: UniPercept can serve as an perceptual-level metric that assesses the quality of outputs from any model producing images, covering three complementary dimensions: IAA, IQA, and ISTA.

<p align="center">
    <img src="asserts/img/metrics_dpg.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/metric_geneval.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
</p>

### UniPercept-Constructed Image Profiles
UniPercept performs comprehensive perceptual-level image analysis, delivering accurate visual ratings across the IAA, IQA, and ISTA dimensions, along with fine-grained multi-dimensional analytical outputs that together form a detailed image profile.


<p align="center">
    <img src="asserts/img/profile1.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/profile2.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/profile3.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
</p>

## ✏️ Citation

If you find UniPercept useful for your research, please consider citing our work:

```
@article{unipercept2026,
  title={UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture},
  author={Anonymous},
  journal={CVPR Submission},
  year={2026}
}
```
