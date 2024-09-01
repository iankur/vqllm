import sys
import time

from typing import Any, Dict

import datasets

import lm_eval
import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table

from omegaconf import DictConfig
from torch import nn
from torchtune import config, utils

from vqllm.utils.eleuther_eval import (
    _EvalWrapper,
    EleutherEvalRecipe as OrigEleutherEvalRecipe,
)


logger = utils.get_logger("DEBUG")

# https://github.com/EleutherAI/lm-evaluation-harness/blob/67a990e7345d4ba940e8281ac7c9113ccef2a446/lm_eval/__main__.py#L365C9-L365C61
logger.info(
    "Some datasets, e.g. hellaswag, may require trust_remote_code to be set to True. "
    "Therefore, setting datasets.config.HF_DATASETS_TRUST_REMOTE_CODE to True for remote execution."
)
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


class EleutherEvalRecipe(OrigEleutherEvalRecipe):
    def setup(self) -> None:
        self._metric_logger = config.instantiate(self._cfg.metric_logger)
        self._metric_logger.log_config(self._cfg)
        super().setup()

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        model = super()._setup_model(model_cfg, model_state_dict)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate(self) -> None:

        model_eval_wrapper = _EvalWrapper(
            self._model,
            self._tokenizer,
            device=self._device,
            max_seq_length=self._cfg.max_seq_length,
            batch_size=self._cfg.batch_size,
            dtype=self._dtype,
            add_bos=self._cfg.add_bos,
        )

        # Task initialization API changed between v0.4.1 and 0.4.2
        try:
            lm_eval.tasks.initialize_tasks()
        except Exception:
            pass

        num_fewshot_list = self._cfg.get("num_fewshot", [None] * len(self._tasks))
        num_fewshot_list = num_fewshot_list + [None] * (
            len(self._tasks) - len(num_fewshot_list)
        )
        for task_name, num_fewshot in zip(self._tasks, num_fewshot_list):
            if num_fewshot is not None:
                logger.info(f"Running {num_fewshot}-shot evaluation on {task_name}.")

            t1 = time.time()
            output = simple_evaluate(
                model=model_eval_wrapper,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=self._limit,
            )
            logger.info(f"Eval completed in {time.time() - t1:.02f} seconds.")

            formatted_output = make_table(output)
            print(formatted_output)

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup()
    recipe.evaluate()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
