import os
import sys
import glob
import json
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import pickle
import datetime
import argparse


def cut_sentence(s, punc=['。','？','！'], maxlen_sentence=120):
    words = s
    sentences = []
    sentence = ''
    ik = 0
    while ik < len(words):
        sentence += words[ik]
        if len(sentence) >= maxlen_sentence or words[ik] in punc:
            if len(sentence) >= maxlen_sentence and words[ik] not in punc:
                found = False
                for ir in range(len(sentence)-1,-1,-1):
                    if sentence[ir] in ['、']:
                        found = True
                        sentences.append(sentence[:ir+1])
                        ik = ik - (len(sentence)-1-ir)
                        sentence = ''
                        break
                if not found:
                    sentences.append(sentence)
                    sentence = ''
            else:
                sentences.append(sentence)
                sentence = ''
            ik += 1
        else:
            ik += 1
    if sentence:
        sentences.append(sentence)
    return [sen for sen in sentences if sen.strip()]


def read_one_file(file_name):
    file_content = []
    with open(file_name,'rt',encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            file_content.append(row)
    return file_content


def parse_content_base(file_content):
    '''

    :param file_content:
    :return:
    '''
    ret = []
    for content in file_content:
        if 'type' not in content:
            continue
        if content['type'] in ['页眉','页脚']:
            continue
        if content['type'] == 'text' and content['inside'] == '':
            continue
        if content['type'] == 'excel':
            ret.append(';'.join(eval(content['inside'])))
        else:
            ret.append(content['inside'])
    return ret


def preprocess(text):
    text = re.sub('\.{6,}','\.',text)


def main(arg):
    processed_text_dir = '/root/autodl-tmp/fintech_data/processed_text/'+arg.version
    embedding_dir = '/root/autodl-tmp/fintech_data/mds_vec_store/'+arg.version
    folder_path = '/root/autodl-tmp/alltxt'
    # process_all_pdfs_in_folder(folder_path)
    file_paths = glob.glob(f'{folder_path}/*')
    file_paths = sorted(file_paths, reverse=True)[:arg.handle_file_num][arg.handle_file_begin_offset:arg.handle_file_end_offset]
    print('file total num：', len(file_paths))

    file2content = defaultdict(list)
    for file_path in tqdm(file_paths):
        # if '核新同花顺' not in file_path:
        #     continue
        fcontent = read_one_file(file_path)
        file2content[file_path] = fcontent
    print('read file total: ', len(file2content))

    print('begin parsing annual report:', datetime.datetime.now())
    t0 = time.time()
    # 解析每个年报的文本内容
    file2contentstr = defaultdict(list)
    for k, v in file2content.items():
        file2contentstr[k].append(parse_content_base(v))

    if not os.path.exists(processed_text_dir):
        os.makedirs(processed_text_dir)
    file2parse_file_path = os.path.join(processed_text_dir, 'file2parse_{}_{}.pkl'.format(arg.handle_file_begin_offset, arg.handle_file_end_offset))
    pickle.dump(file2contentstr, open(file2parse_file_path, 'wb'))
    print('finish parsing annual report:', datetime.datetime.now())

    # 针对每个年报进行分句
    print('begin splitting sentence  on  annual report:', datetime.datetime.now())
    file2row = defaultdict(list)
    file2sent = defaultdict(list)
    for k,v in tqdm(file2contentstr.items()):
        try:
            rows = []
            sents_flat = []
            for sentence in v[0]:
                every_sentence = cut_sentence(sentence, maxlen_sentence=120)
                rows.append(every_sentence)
                sents_flat.extend(every_sentence)
            file2row[k] = rows
            file2sent[k] = sents_flat
            # file2row_file_path = os.path.join(processed_text_dir, k.split('/')[-1].replace('.txt', '_file2row.pkl'))
            # pickle.dump(rows, open(file2row_file_path, 'wb'))
            # file2sent_file_path = os.path.join(processed_text_dir, k.split('/')[-1].replace('.txt', '_file2sent.pkl'))
            # pickle.dump(sents_flat, open(file2sent_file_path, 'wb'))
        except Exception as e:
            print(e)
            print(k)

    file2row_file_path = os.path.join(processed_text_dir,'file2row_{}_{}.pkl'.format(arg.handle_file_begin_offset, arg.handle_file_end_offset))
    pickle.dump(file2row, open(file2row_file_path, 'wb'))
    file2sent_file_path = os.path.join(processed_text_dir,'file2sent_{}_{}.pkl'.format(arg.handle_file_begin_offset, arg.handle_file_end_offset))
    pickle.dump(file2sent, open(file2sent_file_path, 'wb'))
    print('finish splitting sentence  on  annual report:', datetime.datetime.now())

    # 针对每个年报的句子进行embedding
    ## 加载句向量模型
    # model = SentenceModel('/root/autodl-tmp/text2vec-large-chinese')
    model = SentenceTransformer('/root/autodl-tmp/bge-large-zh')
    ### '/root/autodl-tmp/alltxt/2023-06-28__湖南南新制药股份有限公司__688189__南新制药__2021年__年度报告.txt'
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir,exist_ok=True)
    # file2embedding = defaultdict()
    print('begin embedding annual report:', datetime.datetime.now())
    t1 = time.time()
    for k, v in tqdm(file2sent.items()):
        try:
            file_embedding = model.encode(v, normalize_embeddings=True)
            # file2embedding[k] = file_embedding
            embedding_file_path = os.path.join(embedding_dir, k.split('/')[-1].replace('.txt','_embedding.pkl'))
            pickle.dump(file_embedding, open(embedding_file_path,'wb'))
        except Exception as e:
            print(e)
            print(k)
    # file2embedding_file_path = 'v1_embedding.pkl'
    # file2embedding_file_path = os.path.join(embedding_dir, file2embedding_file_path)
    # pickle.dump(file2embedding, open(file2embedding_file_path,'wb'))
    t2 = time.time()
    print('finish embedding annual report:', datetime.datetime.now(),' cost time(s):', t2-t1)
    print('total cost:', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
