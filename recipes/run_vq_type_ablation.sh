# metric logger is set to wandb
WANDB_PROJECT="metric_logger.project=vqllm"
WANDB_GROUP="metric_logger.group=vq_type_ablation"
WANDB_NAME="metric_logger.name"

# eval meta/llama3-8b
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $WANDB_PROJECT $WANDB_NAME="llama3_8b"


# train and eval vqllm/llama3-8b
VQ_KEY="model.vq_attn_key"
VQ_VALUE="model.vq_attn_value"
FAST_QUANTIZER="model.use_fast_quantizer"
RESIDUAL_CODEBOOKS="model.num_residual_codebooks"
CODES="model.num_codebook_entries"
CODE_DIM="model.codebook_entry_dim"
REORDER_CHANNEL="model.vq_attn_key_reorder_channel"

dhat=32
C=2048
K=8

# train and eval with only vq key
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/key"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/llama3_8b_single_device.yaml \
    $VQ_KEY=True $VQ_VALUE=False $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=False checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_key"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $VQ_KEY=True $VQ_VALUE=False $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=False checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=['meta_model_0.pt'] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_key"

# train and eval with only vq key but with channel reordering
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/key_reorder"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/llama3_8b_single_device.yaml \
    $VQ_KEY=True $VQ_VALUE=False $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_key_reorder"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $VQ_KEY=True $VQ_VALUE=False $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=['meta_model_0.pt'] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_key_reorder"

# train and eval with only vq value
# Note: reordering is not applicable for value
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/value"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/llama3_8b_single_device.yaml \
    $VQ_KEY=False $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_value"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $VQ_KEY=False $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=['meta_model_0.pt'] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_value"

# train and eval with vq for both key (with channel reordering) and value
CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/key_reorder_value"

# train
tune run recipes/full_finetune_single_device.py \
    --config recipes/config/llama3_8b_single_device.yaml \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.output_dir=$CKPT_DIR \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_key_reorder_value"

# eval
tune run recipes/eleuther_eval.py \
    --config recipes/config/eleuther_evaluation.yaml \
    $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER=True \
    $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
    $REORDER_CHANNEL=True checkpointer.checkpoint_dir=$CKPT_DIR \
    checkpointer.checkpoint_files=['meta_model_0.pt'] \
    $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_key_reorder_value"
