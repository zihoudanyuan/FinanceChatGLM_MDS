from sqlalchemy import create_engine, MetaData, Table, String, Column, Integer, text, Float
import pandas as pd
from transfer_to_excel import pad_stock_codes

DATABASE_URL = "postgresql://fintech:soft1212@localhost:5432/fintech"

file_df = pd.read_csv('../data/train.csv')
samples = []
for x in file_df['name'].tolist():
    parts = x.split('__')
    parts[-1] = parts[-1].replace('.pdf','')
    samples.append(parts+[x])
# file_df = pd.DataFrame
# simples = [x.split('__')+[x] for x in file_df['name'].tolist()]
pd.DataFrame(samples,columns=['report_date','company_full_name','stock_code','company_short_name','report_year','report_type','file_name']).to_excel('../data/company_annual_reports.xlsx',index=False)

FILE_PATH = "../data/company_annual_reports.xlsx"

df = pd.read_excel(FILE_PATH)
df["stock_code"] = pad_stock_codes(df["stock_code"])
engine = create_engine(DATABASE_URL)
df.to_sql('company_annual_reports', engine, if_exists='replace', index=False,)