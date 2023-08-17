import os
import sys
import glob
import json
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer
import pickle
import datetime
import argparse


def main(arg):
    question_embedding_path = os.path.join(arg.question_embedding_dir, arg.question_embedding_version)
    if not os.path.exists(question_embedding_path):
        os.makedirs(question_embedding_path)
    test_data = []
    with open('/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/test_questions.jsonl','rt',encoding='utf-8') as f:
        for line in f:
            one_data = json.loads(line.strip())
            test_data.append(one_data)
    
    ret = [one_data['question'] for one_data in test_data]
    # for one_data in test_data:
    #     question = one_data['question']
    ## 加载句向量模型
    # model = SentenceModel('/root/autodl-tmp/text2vec-large-chinese')
    instruction = "为这个句子生成表示以用于检索相关文章："
    model = SentenceTransformer('/root/autodl-tmp/bge-large-zh')
    file_embedding = model.encode([instruction+q for q in ret], normalize_embeddings=True)
    # file2embedding[k] = file_embedding
    question_file_path = os.path.join(question_embedding_path, arg.question_embedding_file) # os.path.join(embedding_dir, k.split('/')[-1].replace('.txt','_embedding.pkl'))
    pickle.dump(file_embedding, open(question_file_path,'wb'))

# instruction = "为这个句子生成表示以用于检索相关文章："
# model = SentenceTransformer('BAAI/bge-large-zh')
# q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_embedding_dir", type=str,
                        help="process annual report begin offset")
    parser.add_argument("--question_embedding_version", type=str,
                        help="process annual report end offset")
    parser.add_argument("--question_embedding_file", type=str,
                        help="process annual report end offset")
    parser.add_argument("--version", type=str,
                        help="version of text and embedding store dir")
    parser.add_argument("--handle_file_num", type=int,
                        help="process annual report num")
    parser.add_argument("--handle_file_begin_offset", type=int,
                        help="process annual report begin offset")
    parser.add_argument("--handle_file_end_offset", type=int,
                        help="process annual report end offset")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    main(args)
    # tmp = pickle.load(open('fintech_data/question_embedding/v2/question_embedding_file.pkl','rb'))
    # print(len(tmp))