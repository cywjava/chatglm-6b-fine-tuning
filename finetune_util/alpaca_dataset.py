import json

import torch
import random
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


class AlpacaDataset(Dataset):
    """
        AplacaDataset
        author:chen.yiwan
        date:2023-04-18
    """

    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

        # 定义prompt
        self.PROMPT_PATTERN = "问：{}\n：{}"
        self.SEP_PATTERN = "\n答："
        self.max_src_length = 512
        self.max_dst_length = 512
        # 定义一些常量（chen.yiwan）
        self.bos = tokenizer.bos_token_id
        self.eop = tokenizer.eop_token_id
        self.eos = tokenizer.eos_token_id
        self.unk = tokenizer.unk_token_id
        self.pad = tokenizer.pad_token_id
        self.mask = tokenizer.mask_token_id
        self.gmask = tokenizer.sp_tokenizer[tokenizer.gMASK_token]

    @staticmethod
    def load_json(file_list):
        result_data = []
        for jf in file_list:
            with open(jf, 'r', encoding="utf-8") as file:
                try:
                    json_data = json.load(file)
                except Exception as e:
                    print("load json file,occurring format error, file:", jf)
                    exit(0)
            for item in json_data:
                result_data.append(item)
            random.shuffle(result_data)
        return result_data

    def create_prompt(self, instruction, input):
        return self.PROMPT_PATTERN.format(instruction, input), self.SEP_PATTERN

    def create_prompt_ids(self, instruction_str, input_str):
        prompt, sep = self.create_prompt(instruction=instruction_str, input=input_str)
        sep_ids = self.tokenizer.encode(
            sep,
            add_special_tokens=True
        )
        sep_len = len(sep_ids)
        special_tokens_num = 2
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_src_length - (sep_len - special_tokens_num),
            truncation=True,
            add_special_tokens=False
        )

        return prompt_ids + sep_ids

    def create_inputs_and_labels(self, instruction, input, output, device):
        prompt = self.create_prompt_ids(instruction, input)
        completion = self.tokenizer.encode(
            output,
            max_length=self.max_dst_length,
            truncation=True,
            add_special_tokens=False
        )

        inputs = prompt + completion + [self.eop]
        labels = [-100] * len(prompt) + completion + [self.eop]

        inputs = torch.tensor(inputs, dtype=torch.long, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        return inputs, labels

    def get_attention_mask(self, tokenizer, input_ids, device):
        seq = input_ids.tolist()
        context_len = seq.index(self.bos)
        seq_len = len(seq)
        attention_mask = torch.ones((seq_len, seq_len), device=device)
        attention_mask.tril_()
        attention_mask[..., :context_len] = 1
        attention_mask.unsqueeze_(0)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    def get_position_ids(self, tokenizer, input_ids, device, position_encoding_2d=True):
        seq = input_ids.tolist()
        context_len = seq.index(self.bos)
        seq_len = len(seq)

        mask_token = self.mask if self.mask in seq else self.gmask
        use_gmask = False if self.mask in seq else self.gmask

        mask_position = seq.index(mask_token)

        if position_encoding_2d:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            if not use_gmask:
                position_ids[context_len:] = mask_position
            block_position_ids = torch.cat((
                torch.zeros(context_len, dtype=torch.long, device=device),
                torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
            ))
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            if not use_gmask:
                position_ids[context_len:] = mask_position

        return position_ids

    def __getitem__(self, index):
        item_data = self.data[index]
        instruction_str = item_data["instruction"] if item_data.get("instruction") else ""
        input_str = item_data["input"] if item_data.get("input") else ""
        output_str = item_data["output"] if item_data.get("output") else ""

        input_ids, labels = self.create_inputs_and_labels(
            instruction=instruction_str,
            input=input_str,
            output=output_str,
            device=device
        )
        attention_mask = self.get_attention_mask(self.tokenizer, input_ids, device)
        position_ids = self.get_position_ids(self.tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    def __len__(self):
        return len(self.data)
