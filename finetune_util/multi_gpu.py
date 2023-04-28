import os

import torch


class MultiGPU:
    """
    此版代码多卡训练使用的是老版的模型，模型文件与新版不一样
    需要在fine_tuning_chatglm6b.py中，加载get_peft_model 后调用：MultiGPU().set_model_to_gpus(model)，并且去掉加载模型时的device_map="auto"
    author:chen.yiwan
    date:2022-04-26
    """
    multi_gpu_avaliable = False
    gpus = []

    def __init__(self):
        environ = os.environ
        cuda_visible_devices = environ.get("CUDA_VISIBLE_DEVICES")
        print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
        if torch.cuda.is_available():
            if cuda_visible_devices is None or cuda_visible_devices == "":
                cuda_visible_devices = "0"
            self.gpus = cuda_visible_devices.split(",")
            if len(self.gpus) <= 1:
                self.multi_gpu_avaliable = False
            else:
                self.multi_gpu_avaliable = True

    def get_device_map_dict(self):
        device_map_dict = {'transformer.word_embeddings': 0,
                           'transformer.final_layernorm': 0,
                           'lm_head': 0,
                           'transformer.layers.27': 0}
        num_trans_layers = 27
        gpu_target = 0
        for index in range(num_trans_layers):
            if index % len(self.gpus) != 0:
                gpu_target += 1
            else:
                gpu_target = 0
            device_map_dict[f'transformer.layers.{index}'] = gpu_target
        print(device_map_dict)
        return device_map_dict

    def set_model_to_gpus(self, model):
        if not self.multi_gpu_avaliable:
            return
        else:
            device_map_dict = self.get_device_map_dict()
            model.is_parallelizable = True
            model.model_parallel = True
            for k, v in device_map_dict.items():
                if k == 'transformer.word_embeddings':
                    model.transformer.word_embeddings = model.transformer.word_embeddings.to(f'cuda:{v}')
                if k.find("transformer.layers") != -1:
                    sub_value = int(k.replace("transformer.layers.", ""))
                    model.transformer.layers[sub_value] = model.transformer.layers[sub_value].to(f'cuda:{v}')
                    model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_A = \
                        model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_A.to(f'cuda:{v}')
                    model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_B = \
                        model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_B.to(f'cuda:{v}')
                if k == "transformer.final_layernorm":
                    model.transformer.final_layernorm = model.transformer.final_layernorm.to(f'cuda:{v}')

    def m_gpu_is_avaliable(self):
        return self.multi_gpu_avaliable
