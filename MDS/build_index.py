import faiss
import argparse
import numpy as np
import os
import glob
from tqdm import tqdm
from collections import defaultdict
import pickle

def main(arg):
    embedding_dir = arg.embedding_dir
    index_dir = os.path.join(arg.index_dir,arg.index_version)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    file_paths = glob.glob(f'{embedding_dir}/*')
    file_paths = sorted(file_paths, reverse=True)[:arg.handle_file_num][arg.handle_file_begin_offset:arg.handle_file_end_offset]
    print('embedding file total num：', len(file_paths))

    file2index = defaultdict(list)
    failure_files = []
    for file_path in tqdm(file_paths):
        try:
            x = np.array(pickle.load(open(file_path,'rb')))
            # print(x.shape)
            # n_samples = x.shape[0]
            d = x.shape[1]
            xb = x.astype('float32')
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            index_file_path = os.path.join(index_dir, os.path.basename(file_path).replace('.pkl', '.index'))
            faiss.write_index(index, index_file_path)
            file2index[os.path.basename(file_path)].append(index_file_path)
        except Exception as e:
            print(file_path)
            print(e)
            failure_files.append(file_path)

    print('embedding file total num：', len(file2index))
    # print('read file total: ', len(file2content))
    pickle.dump(file2index, open(os.path.join(index_dir, 'file2index.pkl'), 'wb'))
    pickle.dump(failure_files, open(os.path.join(index_dir, 'failure_files.pkl'), 'wb'))
    print('finish,file2index path', os.path.join(index_dir, 'file2index.pkl'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test_file", type=str, default="/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/test_questions.jsonl")
    parser.add_argument("--embedding_dir", type=str, default="/root/autodl-tmp/fintech_data/mds_vec_store/v2")
    parser.add_argument("--index_dir", type=str, default="/root/autodl-tmp/fintech_data/index")
    parser.add_argument("--index_version", type=str, default="v2")
    parser.add_argument("--handle_file_num", type=int,
                        help="process annual report num")
    parser.add_argument("--handle_file_begin_offset", type=int,
                        help="process annual report begin offset")
    parser.add_argument("--handle_file_end_offset", type=int,
                        help="process annual report end offset")
    arg = parser.parse_args()
    main(arg)