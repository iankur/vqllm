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

Verify setup by calling
```
from vqllm.models.llama3 import llama3_8b
```

### Eval
Download model. Set checkpoint path and tokenizer path in config file. One config file is provided in `vqllm/config/custom_eval_config.yaml`.
```
tune download meta-llama/Meta-Llama-3-8B --output-dir recipes/ckpts/llama3_8b
tune run recipes/eleuther_eval.py --config recipes/config/eleuther_evaluation.yaml
```

### Notes
- EMA embedding sum and cluster size parameters are kept in full precision. However, rest of the model can be in lower precision. So, `model.to(new_dtype)` should be handled carefully.
- Residual quantizer applies straight through estimator after input has been quantized with all the residual codebooks. This is different from existing works. We do this to preserve gradient propagation for commitment losses with all the residual codebooks.
- Similarity and EMA update happen in full precision even for low precision inputs. As a result, we accumulate all the residual commitments losses in full precision and cast to original input precision before returning.
- Torchtune text completion dataset returns input and label sequences, both are identical. Shift happens in the recipe. We use packed dataset, which uses different padding value for input and label sequences. Padding value for label sequence is set to `CROSS_ENTROPY_IGNORE_IDX`. We use this value to create mask to ignore some token embeddings when updating codebook.

### Acknowledgements
- Andrej Karpathy's [vector quantization repo](https://github.com/karpathy/deep-vector-quantization)
- OpenAI's [Triton](https://triton-lang.org/main/index.html) language
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [torchtune](https://github.com/pytorch/torchtune)
