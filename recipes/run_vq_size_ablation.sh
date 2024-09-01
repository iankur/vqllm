# eval meta/llama3-8b
tune run recipes/eleuther_eval.py --config recipes/config/eleuther_evaluation.yaml

# train and eval vqllm/llama3-8b
VQ_KEY="model.vq_attn_key"
VQ_VALUE="model.vq_attn_value"
FAST_QUANTIZER="model.use_fast_quantizer"
RESIDUAL_CODEBOOKS="model.num_residual_codebooks"
CODES="model.num_codebook_entries"
CODE_DIM="model.codebook_entry_dim"

# metric logger is set to wandb
WANDB_PROJECT="metric_logger.project=vqllm"
WANDB_GROUP="metric_logger.group=vq_size_ablation"
WANDB_NAME="metric_logger.name"


for dhat in 32 64; do
    for C in 2048 1024; do
        for K in 8 6 4; do
            CKPT_DIR="/home/ubuntu/vqllm/recipes/ckpts/vq_K${K}_C${C}_dhat${dhat}"

            # train
            tune run recipes/full_finetune_single_device.py \
                --config recipes/config/llama3_8b_single_device.yaml \
                $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER \
                $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
                checkpointer.output_dir=$CKPT_DIR \
                $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="train_K${K}_C${C}_dhat${dhat}"

            # eval
            tune run recipes/eleuther_eval.py \
                --config recipes/config/eleuther_evaluation.yaml \
                $VQ_KEY=True $VQ_VALUE=True $FAST_QUANTIZER \
                $RESIDUAL_CODEBOOKS=$K $CODES=$C $CODE_DIM=$dhat \
                checkpointer.checkpoint_dir=$CKPT_DIR \
                checkpointer.checkpoint_files=['meta_model_0.pt'] \
                $WANDB_PROJECT $WANDB_GROUP $WANDB_NAME="eval_K${K}_C${C}_dhat${dhat}"
        done
    done
done
