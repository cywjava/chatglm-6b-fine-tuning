# coding=utf-8
import json
import os
import random

"""
    生成测试json文件
    author:chen.yiwan
    date:2023-04-08
"""


def generate_alpaca(question_file, answer_file, alpaca_file):
    print("开始生成alpaca...")
    answer = ""
    with open(answer_file, mode='r', encoding='UTF-8') as vf:
        answer = vf.readline()
    train_json = []
    for line in open(question_file, mode='r', encoding='UTF-8'):
        for w_count in range(100):
            train_json.append(
                {'instruction': '' + line.strip() + '', 'input': '', 'output': '' + answer + ''})
    random.shuffle(train_json)
    json_data = json.dumps(train_json, separators=(',', ': '))
    if os.path.exists(alpaca_file):
        os.remove(alpaca_file)
    f = open(alpaca_file, 'w')
    f.write(json_data)
    f.close()
    print("alpaca生成完成...")


if __name__ == "__main__":
    question_file = "C:\\Users\\myoo\\Desktop\\json\\q4.txt"
    answer_file = "C:\\Users\\myoo\\Desktop\\json\\a4.txt"
    alpaca_file = "../data/a4.json"
    generate_alpaca(question_file, answer_file, alpaca_file)
