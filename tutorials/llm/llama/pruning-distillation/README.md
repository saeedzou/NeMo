# Llama 3.1 Pruning and Distillation with NeMo 2.0 Framework

[Llama 3.1](https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/) models, developed by Meta, are open-source large language models that deliver state-of-the-art performance on popular industry benchmarks. Pretrained on over 15 trillion tokens, they support a 128K token context length. These models are available in three sizes: 8B, 70B, and 405B. Each size offers two variants: base pretrained and instruction tuned.

[NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) provides tools to perform teacher fine-tuning, pruning, and distillation on Llama 3.1 to fit your use case.

[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) is a library (referred to as **Model Optimizer**, or **ModelOpt**) comprising state-of-the-art model optimization techniques including [quantization](https://github.com/NVIDIA/TensorRT-Model-Optimizer#quantization-examples-docs), [distillation](https://github.com/NVIDIA/TensorRT-Model-Optimizer#distillation-examples-docs), [pruning](https://github.com/NVIDIA/TensorRT-Model-Optimizer#pruning-examples-docs), and [speculative decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer#speculative-decoding-examples-docs) to compress models.

[LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796) provides details on teacher fine-tuning, pruning, and distillation on Llama 3.1 as described in the [tech report](https://arxiv.org/abs/2408.11796).

[How to Prune and Distill Llama-3.1 8B to an NVIDIA Llama-3.1-Minitron 4B Model](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/) provides practical and effective structured compression best practices for LLMs that combine depth, width, attention, and MLP pruning with knowledge distillation-based retraining.

[Mistral-NeMo-Minitron 8B Model Delivers Unparalleled Accuracy](https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/) introduces the Mistral-NeMo-Minitron 8B, a state-of-the-art 8 billion parameter language model created by pruning and distilling the larger Mistral NeMo 12B model.

## Objectives

This tutorial demonstrates how to perform depth-pruning, width-pruning, teacher fine-tuning, and distillation on **Llama 3.1 8B** using the [WikiText-103-v1](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1) dataset with the NeMo Framework. We will start with a HuggingFace checkpoint and convert it to NeMo format to use for pruning and distillation and later convert the distilled model back to HuggingFace format. The `WikiText-103-v1` language modeling dataset comprises over 100 million tokens extracted from verified Good and Featured articles on Wikipedia.

For this demonstration, we will perform teacher correction by running a light fine-tuning procedure on the `Meta Llama 3.1 8B` teacher model to generate a fine-tuned teacher model, `megatron_llama_ft`, needed for optimal distillation. This fine-tuned teacher model is then trimmed. There are two methods to prune a model: depth-pruning and width-pruning. We will explore both techniques, yielding `3a_depth_pruned_model` and `3b_width_pruned_model`, respectively. These models will serve as starting points for distillation to create the final distilled 4B models `4a_depth_distilled_model` and `4b_width_distilled_model` respectively.

**NOTE:** A subset of functions is being demonstrated in the notebooks. Some features like Neural Architecture Search (NAS) are unavailable, but will be supported in future releases.

## Requirements

### System Configuration
- Access to at least 8 NVIDIA GPUs, each with a memory of at least 80GB (e.g., 8 x H100-80GB or 8 x A100-80GB).
- A Docker-enabled environment, with [NVIDIA Container Runtime](https://developer.nvidia.com/container-runtime) installed, which will make the container GPU-aware.

Get your Hugging Face [access token](https://huggingface.co/docs/hub/en/security-tokens), which will be used to download the Llama 3.1 model and tokenizer.

**NOTE:** The default configuration in the notebook runs on 8 x 80GB NVIDIA GPUs. However, you can potentially reduce the Tensor Parallel size (`TENSOR_PARALLEL_SIZE`) along with the Micro-Batchsize (`MICRO_BATCH_SIZE`) in the teacher fine-tuning and distillation scripts to accommodate lower resource availability.

## Create a Pruned and Distilled Model with NeMo Framework

For pruning and distilling the model, you will use the NeMo Framework, which is available as a [Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo). These notebooks has been tested on `nvcr.io/nvidia/nemo:25.04` container.

**NOTE:** These notebooks use the [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) under the hood for pruning and distillation.

1. Run the container using the following command. You will mount your local directory to `/workspace` so the model and dataset will be stored in a persistent location. If you are using your own model and dataset, you can change the paths in the notebooks accordingly.

```bash
# TODO: Update to 25.04 once the container is released
export FW_VERSION=25.04.rc2
```

```bash
docker run \
  --gpus all \
  --shm-size=16GB \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:$FW_VERSION bash
```

2. From within the container, copy the notebooks to your local directory so changes remain persistent (only if running first time).

```bash
cp -r /opt/NeMo/tutorials/llm/llama/pruning-distillation/* /workspace
```

3. From within the container, login with your Hugging Face token.

```bash
huggingface-cli login --token <YOUR_HF_ACCESS_TOKEN>
```

4. Start the Jupyter lab:

```bash
pip install --upgrade ipywidgets notebook
jupyter lab --ip 0.0.0.0 --port=8888 --allow-root
```

5. Then, navigate to this directory which contains a list of notebooks that cover all the steps to create a distilled 4B model.

This workflow is structured into seven notebooks:
  1. [Prepare the model and dataset](./01_model_and_data_preparation.ipynb)
  2. [Fine-tune the teacher on the dataset](./02_teacher_finetuning.ipynb)
  3. Prune the fine-tuned teacher model to create a student
  - 3a. [Using depth-pruning](./03_a_depth_pruning.ipynb)
  - 3b. [Using width-pruning](./03_b_width_pruning.ipynb)
  4. Distill knowledge from teacher into student
  - 4a. [Using depth-pruned student](./04_a_distilling_depth_pruned_student.ipynb)
  - 4b. [Using width-pruned student](./04_b_distilling_width_pruned_student.ipynb)
  5. [Display the validation loss and post-process checkpoint](./05_post_processing_and_results.ipynb)

> `NOTE:` We are exploring two methods to prune the fine-tuned teacher model: [depth-pruning](./03_a_depth_pruning.ipynb) and [width-pruning](./03_b_width_pruning.ipynb). Per the [tech report](https://arxiv.org/pdf/2408.11796), we can observe that width-pruning generally outperforms depth-pruning while depth pruned model is generally faster so users can choose to perform either [depth-pruning](./03_a_depth_pruning.ipynb) or [width-pruning](./03_b_width_pruning.ipynb) or both methods.


## Results

**NOTE:** This notebook demonstrates the use of the teacher fine-tuning, pruning, and the distillation scripts. These scripts should ideally be run on a multi-node cluster with a larger `GLOBAL_BATCH_SIZE` and `STEPS` to see improvement in the validation loss.

Here are the validation loss plots over 30 steps of running the training step in the distillation script (at the end of [05_post_processing_and_results.ipynb](./05_post_processing_and_results.ipynb)).

<img src="https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/val_loss_depth_pruned_student_distillation.png" width="400px" alt="Validation Loss Plot When Using the Depth-Pruned Model as the Student">

<img src="https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/val_loss_width_pruned_student_distillation.png" width="400px" alt="Validation Loss Plot When Using the Width-Pruned Model as the Student">
