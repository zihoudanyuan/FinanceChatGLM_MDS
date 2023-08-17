# 项目实现思路
基于小打小闹团队开源的baseline（https://github.com/RonaldJEN/FinanceChatGLM/tree/main）开发，新增了文本向量知识库用于补充非计算题回答时使用的信息。构建文本向量知识库的方法是：基于官方开源的pdf抽取出的txt文本文件，进行切句（句子最大长度120）和句子向量化（使用智源开源的BGE embedding模型，具体版本是bge-large-zh，https://huggingface.co/BAAI/bge-large-zh），通过FAISS实现向量化检索来补充问答时使用的context（faiss返回top3，每个命中的结果同时返回上文2句下文3句）。

# 初赛成绩
得分是60.2

# 实验环境
Ubuntu20.04,1个A40卡，数据盘900G，内存80G

项目代码放在目录/root/下面

数据文件放在了数据盘/root/autodl-tmp下面

# 金融数据库的构建
本部分采用小打小闹团队开源的baseline进行构建。
## 数据目录
官方数据集(https://modelscope.cn/datasets/modelscope/chatglm_llm_fintech_raw_dataset/summary) 放到目录 /root/autodl-tmp/chatglm_llm_fintech_raw_dataset

/root/autodl-tmp/alltxt 存放官方开源的txt数据集（wget https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/hackathon_chatglm_fintech/alltxt.zip）

/root/autodl-tmp/processed_middle 存放从html抽取出的年报三表、txt抽取出的年报三表的结果，未进行合并，中间结果

/root/autodl-tmp/final_data/ 存放从年报抽取出的结构化数据，三表会合并html合txt抽取的结果

## 信息抽取

### 01 pdf2html
在 data_extract/01pdf2html/pdf2html目录，先安装jdk和maven，然后通过maven下载依赖，运行 /root/FinanceChatGLM/data_extract/01pdf2html/pdf2html/src/main/java/xfp/pdf/run/Pdf2htmlV2.java，相比baseline，增加了线程池来加快处理速度。

输入目录是 /root/autodl-tmp/chatglm_llm_fintech_raw_dataset/allpdf

输出目录是 /root/autodl-tmp/chatglm_llm_fintech_raw_dataset/html_result
### 02 html提取三表

改写了baseline 的notebook为py脚本， 在 data_extract/02_html提取三表/2019-2021年度,执行下面的命令

conda activate cft

python get_report_info.py

输入目录是 '/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/html_result'

输出目录是 "/root/autodl-tmp/processed_middle/html抽取三表19到21/"

### 03 官方txt提取三表

首先配置go环境，然后在目录 /root/FinanceChatGLM/data_extract/03_官方txt提取三表/2019-2021年/get-report-from-txt 执行

go build

./get-report-from-txt

输入数据目录 "/root/autodl-tmp/alltxt/"

输出数据目录 "/root/autodl-tmp/processed_middle/txt抽取三表19到21/"

### 04 合并html合txt结果，抽取会计科目
本部分对baseline进行了修改，解决了合并时使用的默认状态不一致导致的合并抽取结果为空的问题

首先配置go环境，在目录/root/FinanceChatGLM/data_extract/04_从02和03中抽取会计科目/2019-2021年/get-csv-from-report执行

go build

./get-csv-from-report

// 官方txt抽取出的报表位置
输入数据目录1 "/root/autodl-tmp/processed_middle/txt抽取三表19到21/"

// 我们抽取出的html的位置
输入数据目录2 "/root/autodl-tmp/processed_middle/html抽取三表19到21/"

// 抽取出的csv位置
输出数据目录 "/root/autodl-tmp/final_data/"

### 05 其他基础信息提取

#### 员工信息抽取

首先配置go环境，在目录 /root/FinanceChatGLM/data_extract/05_其他基础信息提取/员工信息/get-employee-info执行

go build

./get-employee-info

输入数据目录 "/root/autodl-tmp/alltxt/"

输出数据 "/root/autodl-tmp/final_data/people.csv"


#### 地址+邮箱+法人+网站信息抽取

首先配置go环境，在目录 /root/FinanceChatGLM/data_extract/05_其他基础信息提取/地址+邮箱+法人+网站/get-report-info3执行

go build

./get-report-info3

输入数据目录 "/root/autodl-tmp/alltxt/"

输出数据 "/root/autodl-tmp/final_data/baseinfo.csv"


#### 股数抽取
首先配置go环境，在目录 /root/FinanceChatGLM/data_extract/05_其他基础信息提取/股数/get-report-info5执行

go build

./get-report-info5

输入文件 "/root/autodl-tmp/alltxt/"


输出文件 "/root/autodl-tmp/final_data/sharesnum.csv"

## 金融数据库构建

### 配置postgresql数据库

本地安装postgresql，创建账号fintech，密码是soft1212，创建数据库名fintech。

### 准备写表的数据

将/root/autodl-tmp/final_data下的所有抽取出的最终数据，复制到目录/root/FinanceChatGLM/llm_demo/data

将官方数据集下的/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/train.csv文件，复制到/root/FinanceChatGLM/llm_demo/data

### 写表
其次，在/root/FinanceChatGLM/llm_demo/util执行下面的写表脚本

conda activate cft

python transfer_to_excel.py

python insert_balance_sheet.py 

python insert_cash_flow.py 

python insert_company_annual_reports.py 

python insert_profit_statement.py

至此，完成金融数据的抽取和数据库表的写入

# 文本向量知识库的构建
## 数据目录
在MDS目录执行下面的代码，在本地创建下面的目录结构

/root/autodl-tmp/alltxt 存放官方开源的txt数据集（wget https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/hackathon_chatglm_fintech/alltxt.zip）

/root/autodl-tmp/fintech_data/processed_text存放切句后的文本数据

/root/autodl-tmp/fintech_data/mds_vec_store 存放句子embedding完的向量

/root/autodl-tmp/fintech_data/question_embedding 存放question embedding完的向量

/root/autodl-tmp/fintech_data/index 存放构建出的faiss索引，每个年报txt文件构建一个索引

## 句子向量化模型
下载BEG句子向量化模型bge-large-zh，放到目录/root/autodl-tmp/bge-large-zh

## python虚拟环境
python虚拟环境构建命令 conda env create -f cft_environment.yml --name cft

## 向量化
通过conda activate cft 激活虚拟环境cft

### 年报切句子&句子embedding

nohup sh parse_split_and_embedding_bge.sh > log_20230807v2.txt 2>&1 &

### 问题embedding

python embedding_question_bge.py --question_embedding_dir=/root/autodl-tmp/fintech_data/question_embedding --question_embedding_version=v3 --question_embedding_file=question_embedding_file.pkl

## 创建Faiss索引

python build_index.py --embedding_dir=/root/autodl-tmp/fintech_data/mds_vec_store/v3  --index_dir=/root/autodl-tmp/fintech_data/index --index_version=v3


# 问答
## 问答使用的大模型
使用官方开源的chatglm2-6b版本（https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary），未进行微调，请下载模型后放置在目录 /root/autodl-tmp/ZhipuAI/chatglm2-6b

## 问答思路

本部主体采用小打小闹团队开源的baseline的问答思路，做的变动是对于非计算题，会先识别问题对应哪份年报（公司+年份），然后从对应年报中检索最相似的3个句子（每个命中的结果同时返回上文2句下文3句），按照句子顺序排序后补充进prompt作为输入。

### 最终测试集预测命令

将官方测试集/root/autodl-tmp/chatglm_llm_fintech_raw_dataset/test_questions.jsonl复制到目录/root/FinanceChatGLM/llm_demo/data目录

通过conda activate cft 激活虚拟环境cft

在llm_demo目录执行下面的命令，获取测试集预测结果

export PYTHONPATH=$PYTHONPATH:/root/FinanceChatGLM/llm_demo/method

export PYTHONPATH=$PYTHONPATH:/root/FinanceChatGLM/llm_demo


python glm_demo_v2.py  --index_dir=/root/autodl-tmp/fintech_data/index --index_version=v3  --question_embedding_dir=/root/autodl-tmp/fintech_data/question_embedding --question_embedding_version=v3 --question_embedding_file=question_embedding_file.pkl  --knowledge_text_version=v3  --question_knowledge_store_file=question_knowledge_store_file.pkl

将预测出的./submit_example.json文件更名为“机器爱学习_result_0812_v2.json”，就得到了提交天池得分最高的结果文件（得分60.2），此提交文件放在了目录/root/FinanceChatGLM/llm_demo/submit/机器爱学习_result_0812_v2.json
