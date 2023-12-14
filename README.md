# M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts

<div align="center">
  <p align="center">
    <a href="https://m3dbench.github.io/">Project Page</a> •
    <a href="https://github.com/OpenM3D/M3DBench">Arxiv Paper</a> •
    <a href="https://github.com/OpenM3D/M3DBench">Dataset</a> •
    <a href="#-citation">Citation
  </p>
  <br>
  <img width="90%" src=./assets/teaser.png>
</div>


## Intro M3DBench

M3DBench introduces a comprehensive 3D instruction-following dataset that encompasses a variety of 3D vision-centric tasks, spanning fundamental abilities in real-world 3D environments.
M3DBench supports multi-modal instructions interleaved with diverse visual prompts.
M3DBench provides a new benchmark for assessing large models across 3D tasks.

## Abstract

Recently, 3D understanding has become popular to facilitate autonomous agents to perform further decisionmaking. However, existing 3D datasets and methods are often limited to specific tasks. On the other hand, recent progress in Large Language Models (LLMs) and Multimodal Language Models (MLMs) have demonstrated exceptional general language and imagery tasking performance. Therefore, it is interesting to unlock MLM’s potential to be 3D generalist for wider tasks. However, current MLMs’ research has been less focused on 3D tasks due to a lack of large-scale 3D instruction-following datasets. In this work, we introduce a comprehensive 3D instructionfollowing dataset called M3DBench, which possesses the following characteristics: 1) It supports general multimodal instructions interleaved with text, images, 3D objects, and other visual prompts. 2) It unifies diverse 3D tasks at both region and scene levels, covering a variety of fundamental abilities in real-world 3D environments. 3) It is a large-scale 3D instruction-following dataset with over 320k instruction-response pairs. Furthermore, we establish a new benchmark for assessing the performance of large models in understanding multi-modal 3D prompts. Extensive experiments demonstrate the effectiveness of our dataset and baseline, supporting general 3D-centric tasks, which can inspire future research.

<!-- <img width="1194" alt="pipeline" src="assets/pipeline.png">
</details> -->

## 🚩 News

- [2023/11/30] Upload paper and init project

## ⚡ Quick Start

<details>
  <summary><b>Setup and download</b></summary>

### 1. Environment

### 2. Data

</details>

## 💻 Train your own models

<details>
  <summary><b>Training</b></summary>
</details>

<details>
  <summary><b>Evaluation</b></summary>
</details>


## Citation

If you find our code or paper helps, please consider starring ⭐ us and citing:



## Acknowledgments

Thanks to [DepthContrast](https://github.com/facebookresearch/DepthContrast), [Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR), [OPT](https://huggingface.co/facebook/opt-6.7b), [Llama 2](https://huggingface.co/meta-llama), and [Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.5). We borrow some of their codes and checkpoints.



## License

This code is distributed under an [MIT LICENSE](LICENSE). If there are any problem regarding our project, please open an issue.
