
<div align="center">

# UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

<a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-UniPercept-red?logo=arxiv" height="25" />
</a>
<a href="https://unipercept.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-UniPercept.github.io-blue" height="25" />
</a>
<a href="https://huggingface.co/collections/unipercept/models" target="_blank">
    <img alt="Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-UniPercept-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/unipercept/unipercept-bench" target="_blank">
    <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-UniPercept--Bench-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

<div style="font-family: charter; text-align: center;">
<a href="#" target="_blank">Shuo Cao</a><sup>*</sup>,
<a href="#" target="_blank">Jiayang Li</a><sup>*</sup>,
<a href="#" target="_blank">Xiaohui Li</a>,
<a href="#" target="_blank">Yuandong Pu</a>,
<a href="#" target="_blank">Kaiwen Zhu</a>,
<a href="#" target="_blank">Yuanting Gao</a>,
<a href="#" target="_blank">Siqi Luo</a>,
<a href="#" target="_blank">Yi Xin</a>,
<a href="#" target="_blank">Qi Qin</a>,
<a href="#" target="_blank">Yu Zhou</a>,
<a href="#" target="_blank">Xiangyu Chen</a>,
<a href="#" target="_blank">Wenlong Zhang</a>,
<a href="#" target="_blank">Bin Fu</a>,
<a href="#" target="_blank">Yu Qiao</a>,
<a href="#" target="_blank">Yihao Liu</a><sup>&#8224;</sup>
<br>
<div style="font-size: 0.9em; margin-top: 0.5em;">
<span>University of Science and Technology of China</span> &emsp;
<span>Shanghai AI Laboratory</span> &emsp;
<span>Peking University</span>
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

## 🚀 News & Updates

- [Dec 17, 2025] 🔥 We release **[UniPercept](https://huggingface.co/unipercept/UniPercept-8B)**, a strong baseline MLLM for perceptual image understanding, empowered by Domain-Adaptive Pre-Training and Task-Aligned RL.
- [Dec 17, 2025] 🔥 We release **[UniPercept-Bench](https://huggingface.co/datasets/unipercept/unipercept-bench)**, a comprehensive perceptual-level MLLM benchmark spanning IAA, IQA, and ISTA domains, supporting both VR and VQA tasks.
- [Dec 17, 2025] 🔥 We release the **[Technical Report](https://arxiv.org/abs/xxxx.xxxxx)** and **[Project Page](https://unipercept.github.io/)**.

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
We introduce **UniPercept-Bench**, a systematic benchmark for perceptual image understanding:
* **Comprehensive Coverage**: Spans **3 domains** (IAA, IQA, ISTA), **17 categories**, and **43 criteria**.
* **Perceptual Tasks**: Supports both **Visual Rating (VR)** and **Visual Question Answering (VQA)**.
<!-- We introduce **UniPercept-Bench**, a systematically designed benchmark for evaluating perceptual-level image understanding.

* **Hierarchical Taxonomy**: Organized into **Domain $\rightarrow$ Category $\rightarrow$ Criterion** layers.
* **Coverage**: 3 Domains (IAA, IQA, ISTA), 17 Categories, and 43 Criterions.
* **Tasks**:
    * **Visual Rating (VR)**: Quantitative scoring of perceptual attributes.
    * **Visual Question Answering (VQA)**: Fine-grained reasoning about visual properties. -->

**Download**: 🤗 [UniPercept-Bench](https://huggingface.co/datasets/unipercept/unipercept-bench)
<p align="center">
    <img src="asserts/img/unipercetp-bench.png" alt="Dataset Distribution" width="1000" height="auto">
</p>

## 🤖 UniPercept

**UniPercept** is a strong baseline MLLM trained via Domain-Adaptive Pre-Training and Task-Aligned RL to handle both **Visual Rating (VR)** (continuous scoring) and **Visual Question Answering (VQA)** (reasoning).

### 🛠️ Setup

```
conda create -n unipercept python=3.10
conda activate unipercept
bash setup.sh
```

### 📉 Evaluation

Please download the UniPercept weights from [🤗 Hugging Face](https://huggingface.co/unipercept/UniPercept-8B) and place them in the `ckpt/` directory.

**Visual Rating (VR)**

Please download the datasets listed below and place them in the corresponding paths.

| Dataset | Domain | Download | Path |
| :--- | :---: | :---: | :--- |
| **ArtiMuse-10K** | IAA | [Link](#) | `benchmark/VR/IAA/ArtiMuse-10K/image` |
| **AVA** | IAA | [Link](#) | `benchmark/VR/IAA/AVA/image` |
| **FLICKR-AES** | IAA | [Link](#) | `benchmark/VR/IAA/FLICKR-AES/image` |
| **PARA** | IAA | [Link](#) | `benchmark/VR/IAA/PARA/image` |
| **TAD66K** | IAA | [Link](#) | `benchmark/VR/IAA/TAD66K/image` |
| **KADID** | IQA | [Link](#) | `benchmark/VR/IQA/KADID/image` |
| **KonIQ-10K** | IQA | [Link](#) | `benchmark/VR/IQA/KonIQ-10K/image` |
| **PIPAL** | IQA | [Link](#) | `benchmark/VR/IQA/PIPAL/image` |
| **SPAQ** | IQA | [Link](#) | `benchmark/VR/IQA/SPAQ/image` |
| **ISTA-10K** | ISTA | [Link](#) | `benchmark/VR/ISTA/ISTA-10K/image` |

After setting up the data, you can configure the target datasets and devices in `src/eval/eval_vr.sh`. The results will be saved to `results/vr`.

```
cd UniPercept
bash src/eval/eval_vr.sh 
```

**Visual Question Answering (VQA)**

Please download **UniPercept-Bench-VQA** from [🤗 Hugging Face](https://huggingface.co/unipercept/UniPercept-8B) and place them into `benchmark/VQA`.
Then you can configure the target domain in `src/eval/eval_vqa.sh`. The evaluation results will be saved to `results/vqa`.

```
cd UniPercept
bash src/eval/eval_vqa.sh 
```

### 🏆 Performance

UniPercept consistently outperforms proprietary models (e.g., GPT-4o, Gemini-2.5-Pro), leading open-source models (InternVL3, Qwen3-VL) and across all three perceptual domains (IAA, IQA, ISTA) and tasks (VR, VQA).

<p align="center">
    <img src="asserts/img/vr.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/vqa-iaa.png" alt="Performance on UniPercept-Bench-VQA (IAA)" width="1000" height="auto">
    <img src="asserts/img/vqa-iqa.png" alt="Performance on UniPercept-Bench-VQA (IQA)" width="1000" height="auto">
    <img src="asserts/img/vqa-ista.png" alt="Performance on UniPercept-Bench-VQA (ISTA)" width="1000" height="auto">
</p>

### 🎨 Applications

**UniPercept As Reward** 

UniPercept can be used as a powerful reward model for post-training Text-to-Image (T2I) models. By integrating UniPercept rewards into the training of **FLUX.1-dev**, we observe significant improvements in aesthetic quality, structural richness, and prompt adherence.

<p align="center">
    <img src="asserts/img/reward.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
</p>

**UniPercept As Metrics**

UniPercept can serve as an perceptual-level metric that assesses the quality of outputs from any model producing images, covering three complementary dimensions: IAA, IQA, and ISTA.

<p align="center">
    <img src="asserts/img/metrics_dpg.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
    <img src="asserts/img/metric_geneval.png" alt="Performance on UniPercept-Bench-VR" width="1000" height="auto">
</p>

### 🖼️ UniPercept-Constructed Image Profiles
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

@misc{cao2025artimusefinegrainedimageaesthetics,
      title={ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding}, 
      author={Shuo Cao and Nan Ma and Jiayang Li and Xiaohui Li and Lihao Shao and Kaiwen Zhu and Yu Zhou and Yuandong Pu and Jiarui Wu and Jiaquan Wang and Bo Qu and Wenhai Wang and Yu Qiao and Dajuin Yao and Yihao Liu},
      year={2025},
      eprint={2507.14533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.14533}, 
}
```
