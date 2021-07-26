# 分数（5）
import json


def read_corpus():

    with open("data/train-v2.0.json", 'r') as load_f:
        load_dict = json.load(load_f)
    datas = load_dict['data']
    qlist = []
    alist = []
    for data in datas:
        for para in data['paragraphs']:
            for qas in para['qas']:
                answers = qas['answers']
                if len(answers) != 0:
                    qlist.append(''.join(qas['question']))
                    for answer in answers:
                        alist.append(''.join(answer['text']))

    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist


