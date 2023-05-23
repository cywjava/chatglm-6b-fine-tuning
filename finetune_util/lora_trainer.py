import os
from typing import Optional

import torch
from transformers import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LoraTrainer(Trainer):
    """
        继承Trainer，重写_save方法，每次save时，保存一个pt
        author:chen.yiwan
        date:2022-04-17 23:42:48
    """

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     return model(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         position_ids=inputs["position_ids"],
    #         labels=inputs["labels"],
    #     ).loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pt_name = "chatglm-6b-lora.pt"

        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        print(f"\nSaving model checkpoint to {output_dir}")
        save_tunable_parameters(
            self.model, os.path.join(output_dir, pt_name)
        )
