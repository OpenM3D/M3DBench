

<div align="center">
  <h1>M3DBench: Let's Instruct Large Models with <br> Multi-modal 3D Prompts</h1>
  <p align="center">
    <a href="https://m3dbench.github.io/">💡Project Page</a> •
    <a href="https://arxiv.org/abs/2312.10763">📃Arxiv Paper</a> •
    <a href="https://huggingface.co/datasets/MSheng-Lee/M3DBench">🗂Dataset</a> •
    <a href="https://github.com/OpenM3D/M3DBench">🤗Checkpoint •
    <a href="#-citation">📖Citation
  </p>
  <br>
  <img width="95%" src=./images/teaser.png>
</div>


## 🏃 Intro M3DBench

M3DBench introduces a **comprehensive** 3D instruction-following dataset with support for **interleaved multi-modal prompts**, covering a variety of fundamental abilities in real-world 3D environments. Furthermore, M3DBench provides a new **benchmark** to assess large models across 3D vision-centric tasks.

<details open="open">
    <summary><b>Abstract</b></summary>

Recently, 3D understanding has become popular to facilitate autonomous agents to perform further decisionmaking. However, existing 3D datasets and methods are often limited to specific tasks. On the other hand, recent progress in Large Language Models (LLMs) and Multimodal Language Models (MLMs) have demonstrated exceptional general language and imagery tasking performance. Therefore, it is interesting to unlock MLM’s potential to be 3D generalist for wider tasks. However, current MLMs’ research has been less focused on 3D tasks due to a lack of large-scale 3D instruction-following datasets. In this work, we introduce a comprehensive 3D instructionfollowing dataset called M3DBench, which possesses the following characteristics: 1) It supports general multimodal instructions interleaved with text, images, 3D objects, and other visual prompts. 2) It unifies diverse 3D tasks at both region and scene levels, covering a variety of fundamental abilities in real-world 3D environments. 3) It is a large-scale 3D instruction-following dataset with over 320k instruction-response pairs. Furthermore, we establish a new benchmark for assessing the performance of large models in understanding multi-modal 3D prompts. Extensive experiments demonstrate the effectiveness of our dataset and baseline, supporting general 3D-centric tasks, which can inspire future research.

</details>

<!-- <img width="1194" alt="pipeline" src="assets/pipeline.png">
</details> -->

## 🚩 News

- [2024/09] **Upload the code** 
- [2024/08] **Release the M3DBench to boost MLLM's 3D perception, reasoning, and planning. See [datasets](https://huggingface.co/datasets/MSheng-Lee/M3DBench).**
- [2024/07] 🎉 **M3DBench is accepted by ECCV 2024**!

### TODO:
<!-- - []  **Upload the code**. -->
- [ ] **Upload training and evaluation scripts**.
- [ ] **Release pre-trained Checkpoint**.
- [ ] **Scale up models**.

## ⚡ Set up

<details>
  <summary><b>Environment Setup</b></summary>

**Step 1. Build Dependencies.** Our code is tested with CUDA 12.2 and Python 3.8.19. It is recommended to create a virtual environment [Optional].

```bash
conda create -n m3dbench python=3.8
conda activate m3dbench
```

Next, you should install the following packages:

```bash
pip install h5py
pip install scipy
pip install cython
pip install plyfile
pip install trimesh==2.35.39
pip install networkx==2.2
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2
pip install numpy==1.19.5
```

After that, build the `pointnet2` and accelerated `giou` from source:

```bash
cd third_party/pointnet2
python setup.py install
```

```bash
cd ../../utils
python cython_compile.py build_ext --inplace
```

</details>



<details>
  <summary><b>Data and Pre-trained Weights Preparation</b></summary>

**Step 1. Prepare the 3D Data and Language Annotations.**

Please refer to the instructions available [here](https://huggingface.co/datasets/MSheng-Lee/M3DBench) to download the pre-processed 3D data and language annotations from M3DBench.

**Step 2. Download Pre-trained weights.** 

You'll need to download the following pre-trained weights for the scene encoder, image encoder, shape encoder, and LLM:

1. **Scene Encoder**  
   We offer two types of 3D scene encoders:  
   - For the PointNet-based encoder, download from [DepthContrast](https://github.com/facebookresearch/DepthContrast).  
   - For the transformer-based encoder, download from [Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR).

2. **Image Encoder**  
   Download the `openai/clip-vit-large-patch14-336` checkpoint (or another image encoder) from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b).

3. **Shape Encoder**  
   Download the pre-trained checkpoint from [3D-VisTA](https://github.com/3d-vista/3D-VisTA).

4. **LLM**

   If your server doesn't support auto-downloading from huggingface, manually download the `meta-llama/Llama-2-7b` checkpoint (or another decoder-only LLM) from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b).

</details>





## 💻 Train your own model

<details>
  <summary><b>Training</b></summary>
</details>

<details>
  <summary><b>Evaluation</b></summary>
</details>


## 📖 Citation

If you find our work helps, please consider starring ⭐ us and citing:

```{bibtex}
@misc{li2023m3dbench,
      title={M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts}, 
      author={Mingsheng Li and Xin Chen and Chi Zhang and Sijin Chen and Hongyuan Zhu and Fukun Yin and Gang Yu and Tao Chen},
      year={2023},
      eprint={2312.10763},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgments

Thanks to [DepthContrast](https://github.com/facebookresearch/DepthContrast), [LL3DA](https://github.com/Open3DA/LL3DA), [CLIP](https://github.com/openai/CLIP), [3D-VisTA](https://github.com/3d-vista/3D-VisTA), [OPT](https://huggingface.co/facebook/opt-6.7b), and [Llama 2](https://huggingface.co/meta-llama). We borrow some of their codes and checkpoints.



## License

This code is distributed under an [MIT LICENSE](LICENSE). If there are any problem regarding our project, please open an issue.
