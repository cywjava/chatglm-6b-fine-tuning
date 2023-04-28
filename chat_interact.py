from transformers import AutoTokenizer, AutoModel
import torch
import sys
import signal

PRE_TRAINED_MODEL_PATH = "../chatglm-6b"


# 退出应用
def app_exit(signum, frame):
    sys.exit()


# 程序入口
def main():
    signal.signal(signal.SIGINT, app_exit)
    signal.signal(signal.SIGTERM, app_exit)
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_PATH, trust_remote_code=True).cuda()
    model.to("cuda")
    history = []
    calc_len = 0
    with torch.autocast("cuda"):
        while True:
            try:
                input_txt = input("user:")
                response, history = model.chat(tokenizer, input_txt, history=history, max_length=calc_len * 10)
                qaTuple = (input_txt, response)
                calc_len = calc_len + len(qaTuple[0]) + len(qaTuple[1])
                history.append(qaTuple)
                print("bot:", response)
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
                break


if __name__ == '__main__':
    main()
