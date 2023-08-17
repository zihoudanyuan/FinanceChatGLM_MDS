import argparse
import json
from method.template_manager import template_manager
from method.prompt_generation_v4 import (
    get_balance_static, get_balance_sheet_prompt, get_profit_statement_prompt,
    get_cash_flow_statement_prompt, calculate_indicator, GLMPrompt
)
from modelscope.utils.constant import Tasks
from modelscope import Model
from modelscope.pipelines import pipeline
from tqdm import tqdm
import pandas as pd
import pickle
import os
from collections import defaultdict
import numpy as np
import faiss
import gc
import sys
from io import StringIO
import contextlib
import re
import timeout_decorator


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


model = Model.from_pretrained('/root/autodl-tmp/ZhipuAI/chatglm2-6b', device_map='auto')
pipe = pipeline(task=Tasks.chat, model=model)
COMPUTE_INDEX_SET = [
    '非流动负债比率', '资产负债比率', '营业利润率', '速动比率', '流动比率', '现金比率', '净利润率',
    '毛利率', '财务费用率', '营业成本率', '管理费用率', "企业研发经费占费用",
    '投资收益占营业收入比率', '研发经费与利润比值', '三费比重', '研发经费与营业收入比值', '流动负债比率'
]

OTHER_TYPE = '其他'

def get_company_info():
    company_df = pd.read_csv('/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/train.csv')
    print('年报总数：', company_df.shape[0])
    simple = [x.split('__')[3] for x in company_df['name'].tolist()]
    complete = [x.split('__')[1] for x in company_df['name'].tolist()]
    company_info = []
    for name in company_df['name'].tolist():
        parts = name.split('__')
        company_info.append({'date': parts[0],'company':parts[1],'code':parts[2],'simple':parts[3],'report_belong_year':parts[4],'report_type':parts[5].rstrip('.pdf')})
    company_info_df = pd.DataFrame(company_info)
    print(company_info_df.head(2))
    return company_info_df


def cls_question_intention(text, company_list, years=['2019','2020','2021']): # '2019年', '2020年', '2021年'
    """
    """
    time_intention = []
    company_intention = []
    for year in years:
        if year in text:
            time_intention.append(year.strip('年'))
    if len(time_intention) == 0:
        time_intention.append(OTHER_TYPE)
    for company_info in company_list:
        if company_info[0] in text: # or company_info[1] in text
            company_intention.append(company_info[0])
            break
    company_intention = list(set(company_intention))
    if len(company_intention) == 0:
        for company_info in company_list:
            if company_info[1] in text:  # 有badcase
                company_intention.append(company_info[0])
                break
    if len(company_intention) == 0:
        company_intention.append(OTHER_TYPE)
    
    return time_intention, list(set(company_intention))


def read_questions(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file.readlines()]
    

# @timeout_decorator.timeout(5)
# def exec_timeout(response_):
#     exec(re.findall('```python([.\s\S]*?)```',response_,re.M)[0].strip())


def process_question(question_obj,submit_file='./submit_example.json',knowledge=None,submit_file_detail='./submit_example_detail.json'):
    glm_prompt = GLMPrompt()

    q = question_obj['question']
    # print('here')
    # print(q)
    contains_year, year_ = glm_prompt.find_years(q)
    stock_name, stock_info, has_stock = glm_prompt.has_stock(q)
    compute_index = False

    if contains_year and has_stock:
        for t in COMPUTE_INDEX_SET:
            if t in q:
                prompt_res = calculate_indicator(year_[0], stock_name, index_name=t)
                if prompt_res is not None:
                    prompt_ = template_manager.get_template("ratio_input").format(context=prompt_res, question=q)
                    inputs_t = {'text': prompt_, 'history': []}
                    response_ = pipe(inputs_t)['response']
                    question_obj["answer"] = str(response_)
                    compute_index = True
                    break

    if not compute_index and '增长率' in q:
        statements = [
            get_profit_statement_prompt(q, stock_name, year_),
            get_balance_sheet_prompt(q, stock_name, year_),
            get_cash_flow_statement_prompt(q, stock_name, year_),
            get_balance_static(q, stock_name, year_)
        ]
        prompt_res = [stmt for stmt in statements if len(stmt) > 5]
        if prompt_res:
            prompt_ = template_manager.get_template("ratio_input").format(context=prompt_res, question=q)
            inputs_t = {'text': prompt_, 'history': []}
            response_ = pipe(inputs_t)['response']
            question_obj["answer"] = str(response_)
            compute_index = True

    if not compute_index:
        prompt_ = glm_prompt.handler_q(question_obj['question'],knowledge,pipe)
        if type(prompt_) is list:
            max_python_run_num = 5
            if '研发人员为' in prompt_[0]:
                max_python_run_num = 10
            run_status = False
            for i in range(max_python_run_num):
                try:
                    inputs_t = {'text': prompt_[0], 'history': []} # 生成python代码
                    response_ = pipe(inputs_t, temperature=0.7)['response']
                    wrong_status = 0
                    # print('############'*2)
                    # print(prompt_[0])
                    # print(response_)
                    # print('############'*3)
                    with stdoutIO() as exec_out:
                        try:
                            # print(response_['response'])
                            # exec(re.findall('```python([.\s\S]*?)```',response_,re.M)[0].strip())
                            if "while True" in response_ or "input" in response_:
                                wrong_status = 1
                                continue
                            else:
                                exec(re.findall('```python([.\s\S]*?)```',response_,re.M)[0].strip())
                            # exec_timeout(response_)
                        except Exception as e1:
                            wrong_status = 1
                            print(str(e1))
                            print("out:", exec_out.getvalue())
                    if wrong_status == 0:
                        question_obj["answer"] = exec_out.getvalue()
                        if '%' in question_obj["answer"]:
                            question_obj["answer"] = question_obj["answer"].replace('%','')
                        run_status = True
                        prompt_ = prompt_[0]
                        # print(exec_out.getvalue())
                        # print('############'*2)
                        break
                except Exception as e:
                    print(e)
            if not run_status:
                # print(f"python calculate wrong:{prompt_[0]}")
                prompt_ = prompt_[1]
                inputs_t = {'text': prompt_[1], 'history': []}
                response_ = pipe(inputs_t)['response']
                question_obj["answer"] = str(response_)
        else:
            # print(prompt_)
            inputs_t = {'text': prompt_, 'history': []}
            response_ = pipe(inputs_t)['response']
            question_obj["answer"] = str(response_)

    with open(submit_file, "a", encoding="utf-8") as f:
        json.dump(question_obj, f, ensure_ascii=False)
        f.write('\n')

    with open(submit_file_detail, "a", encoding="utf-8") as f2:
        question_obj['prompt'] = prompt_
        question_obj['response'] = response_
        json.dump(question_obj, f2, ensure_ascii=False)
        f2.write('\n')

def need_search_in_kg(question_obj):
    glm_prompt = GLMPrompt()

    q = question_obj['question']
    contains_year, year_ = glm_prompt.find_years(q)
    stock_name, stock_info, has_stock = glm_prompt.has_stock(q)
    compute_index = False

    if contains_year and has_stock:
        for t in COMPUTE_INDEX_SET:
            if t in q:
                prompt_res = calculate_indicator(year_[0], stock_name, index_name=t)
                if prompt_res is not None:
                    compute_index = True
                    break

    if not compute_index and '增长率' in q:
        statements = [
            get_profit_statement_prompt(q, stock_name, year_),
            get_balance_sheet_prompt(q, stock_name, year_),
            get_cash_flow_statement_prompt(q, stock_name, year_),
            get_balance_static(q, stock_name, year_)
        ]
        prompt_res = [stmt for stmt in statements if len(stmt) > 5]
        if prompt_res:
            compute_index = True

    if not compute_index:
        return True
    return False

def kg_search(arg):
    # 加载测试集
    test_data = []
    with open('/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/test_questions.jsonl','rt',encoding='utf-8') as f:
        for line in f:
            question_obj = json.loads(line.strip())
            # if need_search_in_kg(question_obj):
            test_data.append(question_obj)
    # #{'id': 0, 'question': '能否根据2020年金宇生物技术股份有限公司的年报，给我简要介绍一下报告期内公司的社会责任工作情况？'}

    test_data = test_data#[:10]

    # 测试集问题意图识别
    company_info_df = get_company_info()
    company_list = company_info_df[['company','simple']].to_numpy().tolist()

    test_data_cls = []
    for one_data in test_data:
        question = one_data['question']
        # company_list = get_company_info(question)
        time_intention, company_intention2 = cls_question_intention(question, company_list)
        # if '康希诺生物股份公司' in question:
        #     print(question)
        # if len(company_intention2) > 1:
        #     print(company_intention2)
        assert len(company_intention2) == 1, (company_intention2, question)
        if company_intention2[0] != OTHER_TYPE:
            assert time_intention[0] != OTHER_TYPE

        # time_company_intentions = [time_intention, company_intention]
        # question_intention = company_intention[0]
        test_data_cls.append({'id': one_data['id'], 'question': question, 'company_intention': company_intention2,'time_intention': time_intention})
    test_data_cls.sort(key=lambda x: x['company_intention'][0])
    test_data_group = defaultdict(list)
    for one_data in test_data_cls:
        test_data_group[one_data['company_intention'][0]].append(one_data)
    # return

    # 测试集问题embedding
    question_embedding_path = os.path.join(arg.question_embedding_dir, arg.question_embedding_version)
    if not os.path.exists(question_embedding_path):
        os.makedirs(question_embedding_path)
    question_embedding_file = os.path.join(question_embedding_path, arg.question_embedding_file) 
    x = np.array(pickle.load(open(question_embedding_file,'rb')))
    question2embedding = defaultdict()
    for question_info, row_idx in zip(test_data, range(x.shape[0])):
        question2embedding[question_info['id']] = x[row_idx,:].reshape(1,-1)
    
       
    # 本地知识库索引库加载
    index_dir = os.path.join(arg.index_dir, arg.index_version)
    file2index = pickle.load(open(os.path.join(index_dir, 'file2index.pkl'), 'rb'))
    intention2index = defaultdict()
    for file_name, index_file_path in file2index.items():
        time_intention, company_intention3  = cls_question_intention(file_name, company_list)
        # assert len(time_intention) == 1
        assert len(company_intention3) == 1,company_intention3
        intention2index[str(time_intention[0])+'_'+str(company_intention3[0])]=index_file_path[0]
    print("索引文件总数：", len(intention2index))
    # tmp_cnt = 0
    # for dk,dv in intention2index.items():
    #     print(dk,dv)
    #     tmp_cnt += 1
    #     if tmp_cnt > 100:
    #         break
    # return

    # 知识库文本加载
    kg_pkl_1 = pickle.load(open(os.path.join('/root/MDS/fintech_data/processed_text/'+arg.knowledge_text_version, 'file2sent_0_6000.pkl'), 'rb'))
    kg_pkl_2 = pickle.load(open(os.path.join('/root/MDS/fintech_data/processed_text/'+arg.knowledge_text_version, 'file2sent_6000_12000.pkl'), 'rb'))
    kg_text_dict = {**kg_pkl_1, **kg_pkl_2}
    intention2kgtext = defaultdict()
    for kg_text_path, kg_sents in kg_text_dict.items():
        time_intention, company_intention  = cls_question_intention(kg_text_path, company_list)
        # assert len(time_intention) == 1
        assert len(company_intention) == 1
        intention2kgtext[str(time_intention[0])+'_'+str(company_intention[0])] = kg_sents
    # return


    # 问答模型加载
    # model = Model.from_pretrained('/root/autodl-tmp/ZhipuAI/chatglm2-6b', device_map='auto')
    # pipe = pipeline(task=Tasks.chat, model=model)

    # embedding_model = SentenceModel('/root/autodl-tmp/text2vec-large-chinese')
    # file_embedding = model.encode(ret)

    # inputs = {'text':'介绍下清华大学', 'history': result['history']}
    # result = pipe(inputs)
    # print(result)
    # {'response': '清华大学是中国著名的高等教育机构之一，位于中国北京市海淀区双清路30号，其历史可以追溯到 1911 年创建的清华学堂，1925 年改名为清华学校，最终在 1949 年成为清华大学。作为我国顶尖的大学之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域具有极高的声誉和影响力。\n\n清华大学的校训是“自强不息、厚德载物”，体现了学校一贯追求的办学理念和核心价值观。清华大学强调“以学生为本”，注重创新教育、素质教育，鼓励学生勇于创新、独立思考，培养学生的综合素质和国际化视野。\n\n清华大学拥有一流的师资队伍，包括众多国内外知名学者和专家，为学生提供了优质的教育资源和良好的学术氛围。校园内设施齐全，包括图书馆、实验室、体育馆、艺术中心等，为学生提供了便利的学习和生活条件。\n\n近年来，清华大学在国内外各个领域取得了巨大的成就，如在科研方面，清华大学的科研团队在多个领域取得了突破性进展，取得了举世瞩目的成果；在人才培养方面，清华大学培养了大批具有国际视野和创新精神的优秀人才，为社会做出了重要贡献。\n\n清华大学是我国乃至全球知名的高等教育机构之一，以其卓越的学术水平和优秀的人才培养成果，成为了国内外各界人士所赞誉的顶尖学府。', 'history': [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('介绍下清华大学', '清华大学是中国著名的高等教育机构之一，位于中国北京市海淀区双清路30号，其历史可以追溯到 1911 年创建的清华学堂，1925 年改名为清华学校，最终在 1949 年成为清华大学。作为我国顶尖的大学之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域具有极高的声誉和影响力。\n\n清华大学的校训是“自强不息、厚德载物”，体现了学校一贯追求的办学理念和核心价值观。清华大学强调“以学生为本”，注重创新教育、素质教育，鼓励学生勇于创新、独立思考，培养学生的综合素质和国际化视野。\n\n清华大学拥有一流的师资队伍，包括众多国内外知名学者和专家，为学生提供了优质的教育资源和良好的学术氛围。校园内设施齐全，包括图书馆、实验室、体育馆、艺术中心等，为学生提供了便利的学习和生活条件。\n\n近年来，清华大学在国内外各个领域取得了巨大的成就，如在科研方面，清华大学的科研团队在多个领域取得了突破性进展，取得了举世瞩目的成果；在人才培养方面，清华大学培养了大批具有国际视野和创新精神的优秀人才，为社会做出了重要贡献。\n\n清华大学是我国乃至全球知名的高等教育机构之一，以其卓越的学术水平和优秀的人才培养成果，成为了国内外各界人士所赞誉的顶尖学府。')]}
 

    # 测试集答案生成
    loaded_indexes = []
    file2loaded_index = defaultdict()
    MAX_LOADED_INDEX_NUM = 100
    test_data_answer = []
    # question_context = defaultdict(list)
    faiss_nprobe = 10
    before_context_num = 2
    after_context_num = 3
    faiss_top_k = 3
    MAX_CONTEXT_LEN = 7200

    # debug_cnt = 0
    miss_cnt = 0

    test_data_knowledge = defaultdict()

    for intention, questions in tqdm(test_data_group.items()): # company_intention
        if intention == OTHER_TYPE:
            miss_cnt += 1
            # continue
            # for question in questions:
            #     model_input = {'text': question['question'], 'history': []}
            #     result = pipe(model_input)
            #     test_data_answer.append({'id': question['id'],'question': question['question'], 'intention': OTHER_TYPE, 'answer': result['response']})
        else:
            # debug_cnt += 1
            # if debug_cnt > 3:
            #     break
            # 加载索引库
            # {'id': one_data['id'], 'question': question, 'company_intention': company_intention,'time_intention': time_intention}
            for question in questions:
                time_intentions = question['time_intention']
                company_intentions = question['company_intention'][0]
                query_indexes = []
                for ti, ci in zip(time_intentions, [company_intentions]*len(time_intentions)):
                    query_indexes.append(str(ti)+'_'+str(ci))
                question_kg_context = []
                question_kg_context_every_one = []
                for query_index in query_indexes:
                    if query_index not in loaded_indexes:
                        if len(loaded_indexes) >= MAX_LOADED_INDEX_NUM:
                            del file2loaded_index[loaded_indexes[0]]
                            loaded_indexes.pop(0)
                            gc.collect()
                        if query_index not in intention2index:  # 没有找到本地索引文件，放弃检索
                            query_parts = query_index.split('_')
                            query_year = int(query_parts[0])
                            if str(query_year+1)+'_'+str(query_parts[1]) in intention2index:
                                query_index = str(query_year+1)+'_'+str(query_parts[1])
                            elif str(query_year+2)+'_'+str(query_parts[1]) in intention2index:
                                query_index = str(query_year+2)+'_'+str(query_parts[1])
                            elif str(query_year-1)+'_'+str(query_parts[1]) in intention2index:
                                query_index = str(query_year-1)+'_'+str(query_parts[1])
                            elif str(query_year-2)+'_'+str(query_parts[1]) in intention2index:
                                query_index = str(query_year-2)+'_'+str(query_parts[1])
                            else:
                                print(query_index," not in intention2index !!!")
                                continue
                        if query_index not in loaded_indexes:
                            loaded_indexes.append(query_index)
                        loaded_faiss_index = faiss.read_index(intention2index[query_index])
                        file2loaded_index[query_index] = loaded_faiss_index
                    question_vector = question2embedding[question['id']]
                    faiss_index = file2loaded_index[query_index]
                    faiss_index.nprobe = faiss_nprobe
                    # if '爱旭股份' in question['question']:
                    #     dummy_q = '沈鸿烈曾在哪里从事博士后研究'
                    #     dummy_q_embedding = embedding_model.encode([dummy_q])
                    #     question_vector = dummy_q_embedding
                    #     print('dummy_q：',dummy_q)
                    _, neighbors = faiss_index.search(question_vector, faiss_top_k)
                    # print(neighbors)
                    if query_index not in intention2kgtext:
                        print(query_index,"not in intention2kgtext")
                        continue

                    neighbor_all_indexes = []
                    for neighbor in neighbors[0]:
                        # print(neighbor,neighbors)
                        neighbor_begin = max(0, neighbor-before_context_num)
                        neighbor_end = min(len(intention2kgtext[query_index]), neighbor+after_context_num)
                        # print(neighbor,neighbor_begin,neighbor_end)
                        for ni in range(neighbor_begin,neighbor_end,1):
                            # print(ni)
                            neighbor_all_indexes.append(ni)
                    # print(neighbor_all_indexes)
                    neighbor_all_indexes = list(set(neighbor_all_indexes))
                    neighbor_all_indexes.sort(reverse=False)
                    # print(neighbor_all_indexes)
                    neighbor_texts = []
                    for neighbor_index in neighbor_all_indexes:
                        # if query_index not in intention2kgtext:
                        #     print(query_index,"not in intention2kgtext")
                        neighbor_texts.append(intention2kgtext[query_index][neighbor_index])
                    # print(neighbor_texts)
                    question_kg_context.extend(neighbor_texts)
                    question_kg_context_every_one.append(neighbor_texts)
                if len(question_kg_context) == 0:
                    continue
                    # print(query_indexes,'find no information in local file')
                    # kg_context = ''
                    # kg_prompt = question['question']
                    # model_input = {'text': kg_prompt, 'history': []}
                    # result = pipe(model_input)
                    # test_data_answer.append({'id': question['id'],'question': question['question'], 'intention': '|'.join(query_indexes), 'answer': result['response'],'final_question': kg_prompt,'kg_context':kg_context,'kg_context_parts':json.dumps(question_kg_context_every_one, ensure_ascii=False)})
                else:
                    # 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
                    kg_context = '\n'.join(question_kg_context)
                    # kg_question = question['question']
                    # if '爱旭股份' in question['question']:
                    #     kg_question = '沈鸿烈曾在哪里从事博士后研究'
                    if len(kg_context) > MAX_CONTEXT_LEN:
                        kg_context = kg_context[:MAX_CONTEXT_LEN]
                    # print(kg_question)
                    # print(kg_context)
#                     kg_prompt = """已知信息：
# {kg_context} 

# 根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{kg_question}""".format(kg_context=kg_context,kg_question=kg_question)
#                     model_input = {'text': kg_prompt, 'history': []}
#                     result = pipe(model_input)
#                     test_data_answer.append({'id': question['id'],'question': kg_question, 'intention': '|'.join(query_indexes), 'answer': result['response'],'final_question': kg_prompt,'kg_context':kg_context,'kg_context_parts':json.dumps(question_kg_context_every_one, ensure_ascii=False)})

                    test_data_knowledge[question['id']] = kg_context
    
    # test_data_answer.sort(key=lambda x: x['id'],reverse=False)
    # with open(arg.submit_file,'wt',encoding='utf8') as f:
    #     for one in test_data_answer:
    #         dummy = {'id' : one['id'],'question': one['question'],'answer': one['answer']}
    #         f.write(json.dumps(dummy, ensure_ascii=False)+'\n')
    # pickle.dump(test_data_answer,open(arg.submit_detail_file,'wb'))
    # print('miss_cnt', miss_cnt)
    # print('debug_cnt', debug_cnt)
    return  test_data_knowledge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Answer question by vector search and semantic search.')
    parser.add_argument("--test_file", type=str, default="/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/test_questions.jsonl")
    parser.add_argument("--index_dir", type=str, default="/root/autodl-tmp/fintech_data/index")
    parser.add_argument("--index_version", type=str, default="v2")
    parser.add_argument("--question_embedding_dir", type=str,
                        help="process annual report begin offset")
    parser.add_argument("--question_embedding_version", type=str,
                        help="process annual report end offset")
    parser.add_argument("--question_embedding_file", type=str,
                        help="process annual report end offset")
    parser.add_argument("--knowledge_text_version", type=str,
                        help="process annual report end offset")
    parser.add_argument("--submit_file", type=str,default='./submit_example.json',
                        help="process annual report end offset")
    parser.add_argument("--submit_detail_file", type=str,
                        help="process annual report end offset")
    parser.add_argument("--question_knowledge_store_file", type=str,
                        help="process annual report end offset")
    arg = parser.parse_args()

    # test_data_knowledge = kg_search(arg)
    # pickle.dump(test_data_knowledge, open(os.path.join(arg.index_dir,arg.index_version+'_'+arg.question_knowledge_store_file), "wb"))
    test_data_knowledge = pickle.load(open(os.path.join(arg.index_dir,arg.index_version+'_'+arg.question_knowledge_store_file), "rb"))
    questions = read_questions("./data/test_questions.jsonl")

    for idx, question_obj in enumerate(tqdm(questions)):
        # if '研发人员占职工' not in question_obj['question']:
        #     continue
        # if '硕士及以上人员占' not in question_obj['question']:
        #     continue

        # print(f'Processing question {idx}\n')
        if question_obj['id'] in test_data_knowledge:
            process_question(question_obj,arg.submit_file,knowledge=test_data_knowledge[question_obj['id']],submit_file_detail=arg.submit_detail_file)
        else:
            process_question(question_obj,arg.submit_file,knowledge=None,submit_file_detail=arg.submit_detail_file)

    # python glm_demo_v2.py  --index_dir=/root/autodl-tmp/fintech_data/index --index_version=v3  --question_embedding_dir=/root/autodl-tmp/fintech_data/question_embedding --question_embedding_version=v3 --question_embedding_file=question_embedding_file.pkl  --knowledge_text_version=v3  --question_knowledge_store_file=question_knowledge_store_file.pkl

    # python glm_demo_v3.py  --index_dir=/root/autodl-tmp/fintech_data/index --index_version=v3  --question_embedding_dir=/root/autodl-tmp/fintech_data/question_embedding --question_embedding_version=v3 --question_embedding_file=question_embedding_file.pkl  --knowledge_text_version=v3  --question_knowledge_store_file=question_knowledge_store_file.pkl --submit_file=./机器爱学习_result_0812_v3.json

    # python glm_demo_v3.py  --index_dir=/root/autodl-tmp/fintech_data/index --index_version=v3  --question_embedding_dir=/root/autodl-tmp/fintech_data/question_embedding --question_embedding_version=v3 --question_embedding_file=question_embedding_file.pkl  --knowledge_text_version=v3  --question_knowledge_store_file=question_knowledge_store_file.pkl --submit_file=./机器爱学习_result_0812_v3.json --submit_detail_file=./机器爱学习_result_0812_v3_detail.json