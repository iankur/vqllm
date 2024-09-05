# model args
MODEL="model._component_"
MODEL_TYPE="checkpointer.model_type"
CHECKPOINTER="checkpointer._component_"
TOKENIZER="tokenizer._component_"
TOKENIZER_PATH="tokenizer.path"

# vq args
VQ_KEY="model.vq_attn_key"
VQ_VALUE="model.vq_attn_value"
FAST_QUANTIZER="model.use_fast_quantizer"
RESIDUAL_CODEBOOKS="model.num_residual_codebooks"
CODES="model.num_codebook_entries"
CODE_DIM="model.codebook_entry_dim"
REORDER_CHANNEL="model.vq_attn_key_reorder_channel"

# metric logger is set to wandb
WANDB_PROJECT="metric_logger.project=vqllm"
WANDB_GROUP="metric_logger.group=vq_model_ablation"
WANDB_NAME="metric_logger.name"

dhat=32
C=2048
K=8

# eval meta/llama3-8b
LLAMA_TOKENIZER=torchtune.models.llama3.llama3_tokenizer
LLAMA_TOKENIZER_PATH=/home/ubuntu/vqllm/recipes/ckpts/llama3_8b/original/tokenizer.model
LLAMA_CHECKPOINTER=vqllm.utils.checkpointer.FullModelMetaCheckpointer

tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.llama3_8b $MODEL_TYPE=LLAMA3 \
    $CHECKPOINTER=$LLAMA_CHECKPOINTER $TOKENIZER=$LLAMA_TOKENIZER \
    $TOKENIZER_PATH=$LLAMA_TOKENIZER_PATH \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="llama3_8b"

# train and eval vqllm/llama3-8b
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/vq_llama3_8b"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/llama3_8b_single_device.yaml \
    $MODEL=vqllm.models.llama3_8b $MODEL_TYPE=LLAMA3 \
    $CHECKPOINTER=$LLAMA_CHECKPOINTER $TOKENIZER=$LLAMA_TOKENIZER \
    $TOKENIZER_PATH=$LLAMA_TOKENIZER_PATH \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_vq_llama3_8b"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.llama3_8b $MODEL_TYPE=LLAMA3 \
    $CHECKPOINTER=$LLAMA_CHECKPOINTER $TOKENIZER=$LLAMA_TOKENIZER \
    $TOKENIZER_PATH=$LLAMA_TOKENIZER_PATH \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=['meta_model_0.pt'] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_vq_llama3_8b"

# eval mistral-7b
MISTRAL_TOKENIZER=torchtune.models.mistral.mistral_tokenizer
MISTRAL_TOKENIZER_PATH=/home/ubuntu/vqllm/recipes/ckpts/mistral_7b/tokenizer.model
MISTRAL_CHECKPOINTER=vqllm.utils.checkpointer.FullModelHFCheckpointer
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/mistral_7b"
MISTRAL_CKPT="[pytorch_model-00001-of-00002.bin,pytorch_model-00002-of-00002.bin]"

tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.mistral_7b $MODEL_TYPE=MISTRAL \
    $CHECKPOINTER=$MISTRAL_CHECKPOINTER checkpointer.checkpoint_dir=$CKPT_DIR \
    $TOKENIZER=$MISTRAL_TOKENIZER $TOKENIZER_PATH=$MISTRAL_TOKENIZER_PATH \
    checkpointer.checkpoint_files=$MISTRAL_CKPT \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="mistral_7b"

# train and eval vqllm/mistral-7b
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/vq_mistral_7b"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/mistral_7b_single_device.yaml \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_vq_mistral_7b"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.mistral_7b $MODEL_TYPE=MISTRAL \
    $CHECKPOINTER=$MISTRAL_CHECKPOINTER $TOKENIZER=$MISTRAL_TOKENIZER \
    $TOKENIZER_PATH=$MISTRAL_TOKENIZER_PATH \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=[hf_model_0001_0.pt,hf_model_0002_0.pt,hf_model_0003_0.pt] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_vq_mistral_7b"

# eval gemma-7b
GEMMA_TOKENIZER=torchtune.models.gemma.gemma_tokenizer
GEMMA_TOKENIZER_PATH=/home/ubuntu/vqllm/recipes/ckpts/gemma_7b/tokenizer.model
GEMMA_CHECKPOINTER=vqllm.utils.checkpointer.FullModelHFCheckpointer
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/gemma_7b"
GEMMA_CKPT="[pytorch_model-00001-of-00002.bin,pytorch_model-00002-of-00002.bin]"

tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.gemma_7b $MODEL_TYPE=GEMMA \
    $CHECKPOINTER=$GEMMA_CHECKPOINTER checkpointer.checkpoint_dir=$CKPT_DIR \
    $TOKENIZER=$GEMMA_TOKENIZER $TOKENIZER_PATH=$GEMMA_TOKENIZER_PATH \
    checkpointer.checkpoint_files=$GEMMA_CKPT \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="gemma_7b"

# train and eval vqllm/gemma-7b
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/vq_gemma_7b"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/gemma_7b_single_device.yaml \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_vq_gemma_7b"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $MODEL=vqllm.models.gemma_7b $MODEL_TYPE=GEMMA \
    $CHECKPOINTER=$GEMMA_CHECKPOINTER $TOKENIZER=$GEMMA_TOKENIZER \
    $TOKENIZER_PATH=$GEMMA_TOKENIZER_PATH \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=[hf_model_0001_0.pt,hf_model_0002_0.pt,hf_model_0003_0.pt] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_vq_gemma_7b"
