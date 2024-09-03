# vqllm
Residual vector quantization for KV cache compression in large language model

## Setup
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

conda create -n vqllm python=3.11
conda activate vqllm

git clone https://github.com/iankur/vqllm.git
cd vqllm
pip install -e .
```

For development env
```
pip install -e .[dev]
pre-commit install
```

Log into huggingface and wandb to download models and save results
```
huggingface-cli login
wandb login
```

### Experiment
All experiments can be launched with the following commands. Note that VQ size and type ablations use llama3 model whereas model ablations uses all the models downloaded below.
```
tune download meta-llama/Meta-Llama-3-8B --output-dir recipes/ckpts/llama3_8b
tune download mistralai/Mistral-7B-v0.1 --output-dir recipes/ckpts/mistral_7b

bash recipes/run_vq_size_ablation.sh
bash recipes/run_vq_type_ablation.sh
bash recipes/run_vq_model_ablation.sh
```

### Notes
- EMA embedding sum and cluster size parameters are kept in full precision. However, rest of the model can be in lower precision. So, `model.to(new_dtype)` should be handled carefully.
- Similarity and EMA update happen in full precision even for low precision inputs. As a result, we accumulate all the residual commitments losses in full precision and cast to original input precision before returning.
- Currently, k-means based initialization uses CPU since GPU based implementations may OOM for ~100K samples. There is minor performance difference between using say ~10K samples vs ~100K samples for the initialization.
- Although seed is set, there seems to be some randomness in current implementation and same setting can lead to small difference in final performance on old-llm-leaderboard tasks of lm-evaluation-harness across multiple runs.
- Torchtune text completion dataset returns input and label sequences, both are identical. Shift happens in the recipe. We modify it to return the actual input and target sequence. We also use packed dataset, which uses different padding value for input and label sequences. Padding value for label sequence is set to `CROSS_ENTROPY_IGNORE_IDX`. We use this value to create vq mask to ignore some token embeddings when updating codebook.

### Acknowledgements
- Andrej Karpathy's [vector quantization repo](https://github.com/karpathy/deep-vector-quantization)
- OpenAI's [Triton](https://triton-lang.org/main/index.html) language
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [torchtune](https://github.com/pytorch/torchtune)
