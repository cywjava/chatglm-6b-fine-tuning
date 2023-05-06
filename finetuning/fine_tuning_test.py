import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel

"""
    测试微调后的AI模型
    author:chen.yiwan
    date:2023-03-31
"""
model_path = "G:\\idea_work2\\chatglm-6b-fine-tuning\\model"
lora_path = "F:\\checkpoints\\chatglm-6b-lora.pt"
# torch.set_default_dtype(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value'],
)
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(lora_path), strict=False)
model.to("cuda")
print("---------------键入:CLS 清空对话信息，EXIT 退出程序!---------------")
history = []
with torch.autocast("cuda"):
    while True:
        try:
            input_txt = input("user:")
            if input_txt == "CLS":
                history = []
                print("已清空聊天数据！")
                continue
            elif input_txt == "EXIT":
                print("退出程序！")
                exit(0)
            response, his = model.chat(tokenizer, input_txt, history=history, max_length=1024)
            if len(history) >= 2:
                history.__delitem__(0)
            history.append(his[len(his)-1])
            print("bot:", response)
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            break
