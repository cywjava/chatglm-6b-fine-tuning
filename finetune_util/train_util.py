import torch


class TrainUtil:
    def __init__(self, run_args, model, tokenizer):
        self.model = model
        self.run_args = run_args
        self.tokenizer = tokenizer

    def resize_tensor(self, type, old_tensor, max_size, padding):
        """
        这尼玛写得对不对也不清楚，晚上跑一下看看效果吧
        :param type:
        :param old_tensor:
        :param max_size:
        :param padding:
        :return:
        """
        if type == "input_ids" or type == "labels":
            new_tensor = torch.full([max_size], padding, dtype=torch.long)
            new_tensor[0:len(old_tensor)] = old_tensor
            return new_tensor
        elif type == "attention_mask":
            new_tensor = []
            attention_mask = old_tensor
            attention_mask = attention_mask.squeeze(-3)
            new_tensor = torch.ones((max_size, max_size), dtype=torch.long)
            new_tensor[0][0:len(attention_mask[0])] = attention_mask[0]
            new_tensor[1][0:len(attention_mask[1])] = attention_mask[1]
            new_tensor.tril_()
            new_tensor.unsqueeze_(0)
            new_tensor = (new_tensor < 0.5).bool()
            return new_tensor
        elif type == "position_ids":
            new_tensor = torch.zeros(2, max_size, dtype=torch.long)
            new_tensor[:, :old_tensor.size()[-1]] = old_tensor
            return new_tensor

    def data_collator(self, features: list) -> dict:
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids) + 1
        input_ids = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = self.resize_tensor("input_ids", feature["input_ids"], longest, self.tokenizer.pad_token_id)
            input_ids.append(ids)
            labels = self.resize_tensor("labels", feature["labels"], longest, -100)
            labels_list.append(labels)
            attention_mask, position_ids = feature["attention_mask"], feature["position_ids"]
            attention_mask = self.resize_tensor("attention_mask", attention_mask, longest, False)
            attention_mask_list.append(attention_mask)
            position_ids = self.resize_tensor("position_ids", position_ids, longest, 1)
            position_ids_list.append(position_ids)

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        position_ids = torch.stack(position_ids_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def print_debug(self):
        if self.run_args.debug:
            print(self.model)
            print(self.tokenizer)
            print("BOS:", self.tokenizer.bos_token_id)
            print("EOP:", self.tokenizer.eop_token_id)
            print("EOS:", self.tokenizer.eos_token_id)
            print("UNK:", self.tokenizer.unk_token_id)
            print("PAD:", self.tokenizer.pad_token_id)
            print("MASK:", self.tokenizer.mask_token_id)
            print("gMASK:", self.tokenizer.sp_tokenizer[self.tokenizer.gMASK_token])
            # 测试
            print("add_special_tokens=True,", self.tokenizer.encode("陈一万", add_special_tokens=True))
            print("add_special_tokens=False,", self.tokenizer.encode("陈一万", add_special_tokens=False))
            print("87604 decode:", self.tokenizer.decode([87604]))
            print("94969 decode:", self.tokenizer.decode([94969]))

            # 另外，当 add_special_tokens = True 时，编码结果会在末尾添加 150001和 150004，
            # 也就是 gmask 和 bos。请注意，我们的训练数据，要按照如下编码要求进行构造：
            # [token, ..., token, gmask, bos, token, ... token, eop]
