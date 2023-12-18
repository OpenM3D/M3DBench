<div align="center">
  <h3> <a href="https://m3dbench.github.io/">M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts</a></h3>
  <p align="center">
    <a href="https://m3dbench.github.io/">Project Page</a> •
    <a href="https://github.com/OpenM3D/M3DBench">Arxiv Paper</a> •
    <a href="https://github.com/OpenM3D/M3DBench">Dataset</a> •
    <a href="#citation">Citation
  </p>
  <br>
  <img width="95%" src=./assets/teaser.png>
</div>


## Intro M3DBench

M3DBench introduces a **comprehensive** 3D instruction-following dataset with support for **interleaved multi-modal prompts**, covering a variety of fundamental abilities in real-world 3D environments. Furthermore, M3DBench provides a new **benchmark** to assess large models across 3D vision-centric tasks.

<details open="open">
    <summary><b>Abstract</b></summary>

Recently, 3D understanding has become popular to facilitate autonomous agents to perform further decisionmaking. However, existing 3D datasets and methods are often limited to specific tasks. On the other hand, recent progress in Large Language Models (LLMs) and Multimodal Language Models (MLMs) have demonstrated exceptional general language and imagery tasking performance. Therefore, it is interesting to unlock MLM’s potential to be 3D generalist for wider tasks. However, current MLMs’ research has been less focused on 3D tasks due to a lack of large-scale 3D instruction-following datasets. In this work, we introduce a comprehensive 3D instructionfollowing dataset called M3DBench, which possesses the following characteristics: 1) It supports general multimodal instructions interleaved with text, images, 3D objects, and other visual prompts. 2) It unifies diverse 3D tasks at both region and scene levels, covering a variety of fundamental abilities in real-world 3D environments. 3) It is a large-scale 3D instruction-following dataset with over 320k instruction-response pairs. Furthermore, we establish a new benchmark for assessing the performance of large models in understanding multi-modal 3D prompts. Extensive experiments demonstrate the effectiveness of our dataset and baseline, supporting general 3D-centric tasks, which can inspire future research.

</details>

<!-- <img width="1194" alt="pipeline" src="assets/pipeline.png">
</details> -->

## News

- [2023/12/15] Upload paper and initialize the project page

## Set up

<details>
  <summary><b>Environment</b></summary>
</details>

<details>
  <summary><b>Data</b></summary>
</details>



## Train your own model

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
