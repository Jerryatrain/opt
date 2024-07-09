import os
import logging

# 项目参数
# end_date = '2017-03-31'  # 必须是交易日
# end_date = '2023-12-29'  # 必须是交易日
# end_date = '2024-01-31'  # 必须是交易日
# end_date = '2024-03-01'  # 必须是交易日
end_date = '2024-06-21'  # 必须是交易日
select_end_date = '2022-12-30'  # 因子筛选结束日期，用于挑中性化barra
# select_end_date = '2024-03-29'  # 因子筛选结束日期，用于挑中性化barra

# end_date = '2023-09-5'


# 默认参数，项目运行后不再更改，否则在自动更新concat逻辑中可能会缺失前值
data_init_date = '2012-12-31'  # 底层数据拉取起始日期，因为是大于号拉取，因此选用上一年年末最后一天
factor_init_date = '2014-12-31'  # 因子计算起始日期，因为是大于号拉取，因此选用上一年年末最后一天
# test_init_date = '2016-12-31'  # 回测起始日期
# test_init_date = '2021-12-31'  # 回测起始日期
test_init_date = '2024-02-29'  # 回测起始日期
benchmark = '000852'  # 基准
# benchmark = '000905'  # 基准


database_path = 'Database/Origin_Database'  # 底层原始数据
neut_data = 'Database/Neut_Database'  # 中性化后的数据
model_path = 'Database/Model_Save'  # 保存临时模型的位置

# 核心文件名称
stock_price_data = 'stock_price.pkl'
barra_factor_data = 'barra_factor.pkl'
industry_cate_data = 'industry_cate.pkl'
industry_price = 'industry_price.pkl'
index_weight = 'index_weight.pkl'
windA_price = 'windA_price.pkl'
index_price = 'index_price.pkl'
north_holding = 'north_holding.pkl'
mkt_capt_data = 'mkt_capt.pkl'
st_signal = 'ST_signal.pkl'
sub_new_signal = 'sub_new_signal.pkl'
quarter_fin = 'quarter_fin.pkl'
quarter_asset = 'quarter_asset.pkl'
fina_ann = 'fina_ann.pkl'

# 读取数据库纪录报错部分
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = 'findata_service'
ROOT_DIR = ROOT_DIR[:(ROOT_DIR.find(PROJECT) + len(PROJECT))]
FailureRetryTimes = 5
LOG_FILE_NAME = os.path.join(ROOT_DIR, 'log', 'findata_service.log')
LOG_FORMAT = '%(asctime)s %(levelname)s %(funcName)s(%(lineno)d): %(message)s'
loggingConfig = {
    'level': logging.INFO,
    'format': LOG_FORMAT,
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'filename': LOG_FILE_NAME,
    'filemode': 'a'
}
