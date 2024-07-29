import time
import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import job_config as jc
import model_config as mc
from joblib import Parallel, delayed
# from Config import factor_config as fc
# from Data_Update.data_update import update_data_net, update_data_local
# from Functions.neut_function import get_neut_factor, get_factor_corr_barra
# from Functions.combine_function import factor_combine, factor_combine_daily, get_adjusted_date
# from Functions.backtest_optimize_function import backtest_optimize
from backtest_single import port_opt, backtest_single, backtest_single_cov, backtest_single_index
import warnings

warnings.filterwarnings('ignore')
# from WindPy import w
# w.start()


def backtest_job():


    # 1、更新底层数据
    # a = time.time()
    # update_data_net(jc.end_date)  # 6.73小时corr_model
    # print('读内网数据', time.time() - a)
    # update_data_local(jc.end_date)  # 本地计算
    # print('读所有因子数据', time.time() - a)

    # 更新均值

    # 1.5获取相关性barra因子
    # get_factor_corr_barra()

    # # 2、更新中性化因子
    # a = time.time()
    # get_neut_factor(jc.end_date)  # 单因子计算耗时全历史耗时约1分钟
    # print('中性化', time.time() - a)
    #
    # # 3、因子合成
    # a = time.time()
    # score_df = factor_combine()  # 3.87小时
    # score_df.to_pickle('score_510.pkl')
    # score_df.to_pickle('score_df_new.pkl')
    # score_df = pd.read_pickle('score_df.pkl')
    # score_df = factor_combine()  # 3.87小时
    # score_df.to_pickle('eps20成交_score_df.pkl')
    # # score_df = pd.read_pickle('score_df.pkl')
    # print('因子合成', time.time() - a)

    # 4、同时回测+组合优化
    # 先做单周期测试
    # score_df = pd.read_pickle('score_df_new.pkl')
    # pool = pd.read_pickle('Database/Origin_Database/stock_pool.pkl')
    # pool_old = pd.read_pickle('Database/Origin_Database/stock_pool_old.pkl')
    # for date_, df_ in pool.groupby('trade_date'):
    #     if date_ > pd.to_datetime('2017-01-01'):
    #         score_df_ = score_df[score_df['trade_date'] == date_]
    #         pool_ = pool_old[pool_old['trade_date'] == date_]
    #         print(date_, len(df_), len(score_df_), len(pool_))
    # score_df = score_df[score_df['security_code'] != '000666']

    # a = time.time()
    # for num in mc.T_pre_list:
    #     score_df_tmp = score_df[['security_code', 'trade_date', 'score_%s' % num]]
    #     score_df_tmp['score_%s_mean' % num] = score_df.groupby('trade_date')['score_%s' % num].transform('mean')
    #     score_df_tmp['score_%s_std' % num] = score_df.groupby('trade_date')['score_%s' % num].transform('std')
    #     score_df_tmp['score_%s' % num] = (score_df_tmp['score_%s' % num] - score_df_tmp['score_%s_mean' % num]) \
    #                                      / score_df_tmp['score_%s_std' % num]
    #     backtest_single(score_df_tmp, turnover_punish=mc.turnover_punish,
    #                     file_name='500单周期拟合回测_惩罚%s_敞口%s_score%s' % (mc.turnover_punish, mc.barra_std_num, num))
    #
    # score_df = pd.read_pickle('score_2model.pkl')
    # col = 'MLP_5'
    # score_df_tmp = score_df[['security_code', 'trade_date', col]]
    # score_df_tmp['%s_mean' % col] = score_df.groupby('trade_date')[col].transform('mean')
    # score_df_tmp['%s_std' % col] = score_df.groupby('trade_date')[col].transform('std')
    # score_df_tmp['score_%s' % col] = (score_df_tmp[col] - score_df_tmp['%s_mean' % col]) \
    #                                  / score_df_tmp['%s_std' % col]
    # backtest_single(score_df_tmp, turnover_punish=mc.turnover_punish,
    #                 file_name='1000单周期拟合回测_惩罚%s_敞口%s_%s' % (mc.turnover_punish, mc.barra_std_num, col))

    score_df = pd.read_pickle('score_2model.pkl')
    col_list = list(score_df.columns)[2:]
    for col in col_list:
        score_df_tmp = score_df.copy()
        score_df_tmp['%s_mean' % col] = score_df.groupby('trade_date')[col].transform('mean')
        score_df_tmp['%s_std' % col] = score_df.groupby('trade_date')[col].transform('std')
        score_df_tmp['score_%s' % col] = (score_df_tmp[col] - score_df_tmp['%s_mean' % col]) \
                                         / score_df_tmp['%s_std' % col]
        score_df['score_%s' % col] = score_df_tmp['score_%s' % col]

    score_df_tmp = score_df[
        ['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22',
         'score_Xgb_22']]
    chosen_num = score_df_tmp.shape[1] - 2
    score_df_tmp['score_mean'] = score_df_tmp.iloc[:, -chosen_num:].mean(axis=1)
    score_df_tmp = score_df_tmp[['security_code', 'trade_date', 'score_mean']]
    jc.benchmark = '000852'
    mc.turnover_punish = 100
    mc.barra_std_num = 0.5
    backtest_single(score_df_tmp, turnover_punish=mc.turnover_punish,
                    file_name='双模型_周期5+10+22_1000单周期拟合回测_惩罚%s_敞口%s' % (
                    mc.turnover_punish, mc.barra_std_num))

    score_df_tmp = score_df[
        ['security_code', 'trade_date', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22']]
    chosen_num = score_df_tmp.shape[1] - 2
    score_df_tmp['score_mean'] = score_df_tmp.iloc[:, -chosen_num:].mean(axis=1)
    score_df_tmp = score_df_tmp[['security_code', 'trade_date', 'score_mean']]
    jc.benchmark = '000300'
    mc.turnover_punish = 300
    mc.barra_std_num = 0.1
    backtest_single(score_df_tmp, turnover_punish=mc.turnover_punish,
                    file_name='双模型_周期10+22_300单周期拟合回测_惩罚%s_敞口%s' % (
                    mc.turnover_punish, mc.barra_std_num))

    score_df_tmp = score_df[
        ['security_code', 'trade_date', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22']]
    chosen_num = score_df_tmp.shape[1] - 2
    score_df_tmp['score_mean'] = score_df_tmp.iloc[:, -chosen_num:].mean(axis=1)
    score_df_tmp = score_df_tmp[['security_code', 'trade_date', 'score_mean']]
    jc.benchmark = '000300'
    mc.turnover_punish = 300
    mc.barra_std_num = 0.3
    backtest_single(score_df_tmp, turnover_punish=mc.turnover_punish,
                    file_name='双模型_周期10+22_300单周期拟合回测_惩罚%s_敞口%s' % (
                    mc.turnover_punish, mc.barra_std_num))


def backtest_job_multi():
    
    score_df = pd.read_pickle('score_2model.pkl')
    col_list = list(score_df.columns)[2:]
    for col in col_list:
        score_df_tmp = score_df.copy()
        score_df_tmp['%s_mean' % col] = score_df.groupby('trade_date')[col].transform('mean')
        score_df_tmp['%s_std' % col] = score_df.groupby('trade_date')[col].transform('std')
        score_df_tmp['score_%s' % col] = (score_df_tmp[col] - score_df_tmp['%s_mean' % col]) \
                                         / score_df_tmp['%s_std' % col]
        score_df['score_%s' % col] = score_df_tmp['score_%s' % col]


    def process_and_backtest(columns, benchmark, turnover_punish, barra_std_num, file_name_suffix):
        # Subset and process the DataFrame
        score_df_tmp = score_df[columns]
        chosen_num = score_df_tmp.shape[1] - 2
        score_df_tmp['score_mean'] = score_df_tmp.iloc[:, -chosen_num:].mean(axis=1)
        score_df_tmp = score_df_tmp[['security_code', 'trade_date', 'score_mean']]
        
        # Set global configuration values
        jc.benchmark = benchmark
        mc.turnover_punish = turnover_punish
        mc.barra_std_num = barra_std_num
        
        # Run the backtest function
        backtest_single(score_df_tmp, turnover_punish=turnover_punish,
                        file_name='双模型_{}单周期拟合回测_惩罚{}_敞口{}'.format(file_name_suffix, turnover_punish, barra_std_num))

    # Define the parameter combinations
    # params = [
    #     (['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22'], '000852', 100, 0.1, '周期5+10+22_1000'),
    #     (['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22'], '000852', 100, 0.3, '周期5+10+22_1000'),
    #     (['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22'], '000852', 100, 0.5, '周期5+10+22_1000'),
    #     (['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22'], '000852', 100, 0.7, '周期5+10+22_1000')
    # ]
    params = []
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        params.append((['security_code', 'trade_date', 'score_MLP_5', 'score_Xgb_5', 'score_MLP_10', 'score_Xgb_10', 'score_MLP_22', 'score_Xgb_22'], '000852', 100, i, '周期5+10+22_1000'))

    # Execute the function in parallel
    results = Parallel(n_jobs=-1)(delayed(process_and_backtest)(*p) for p in params)


def daily_job(end_date):
    # 1、更新底层数据
    a = time.time()
    print('开始计算%s日数据' % end_date)
    update_data_net(end_date)  # 连内网的数据
    print('读内网数据总用时%s分钟' % int((time.time() - a) / 60))
    update_data_local(end_date)  # 本地计算
    print('读所有因子数据总用时%s分钟' % int((time.time() - a) / 60))

    # 1.5获取相关性barra因子，笔记本不进行该更新，因为时间短，需要在电脑端更新
    # get_factor_corr_barra()

    # 2、更新中性化因子
    get_neut_factor(end_date)  # 单因子计算耗时全历史耗时约1分钟
    print('中性化完成总用时%s分钟' % int((time.time() - a) / 60))

    # 3、获取下一个交易日日期
    trade_date = pd.to_datetime(w.tdaysoffset(1, end_date, "").Data[0][0]).strftime('%Y-%m-%d')

    # 4、因子合成
    score_df = factor_combine_daily(end_date, trade_date)
    score_df.to_pickle('daily_result/%s日股票预测得分.pkl' % end_date)
    print('%s合成完成总用时%s分钟' % (end_date, int((time.time() - a) / 60)))


def cal_factor_attribution(past_holding, index_weight, end_date):
    # 读取价格数据
    stock_df = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    trade_date_list = [pd.to_datetime(date_) for date_ in stock_df['trade_date'].unique()]
    stock_df['past_close'] = stock_df.groupby('security_code')['s_dq_close'].shift(1)
    stock_df['chg'] = stock_df['s_dq_close'] / stock_df['past_close'] - 1
    stock_df_today = stock_df[stock_df['trade_date'] == end_date]
    # 读取barra数据
    barra_all = pd.read_pickle('Database/Origin_Database/barra_all.pkl')
    barra_all = barra_all[barra_all['trade_date'] == end_date]
    barra_return = pd.read_pickle('Database/Origin_Database/barra_return.pkl')
    barra_return = barra_return[barra_return['trade_date'] == end_date]
    barra_return.dropna(axis=1, inplace=True)  # 删去已经没有的行业
    stock_barra_return = pd.read_pickle('Database/Origin_Database/stock_barra_return.pkl')
    stock_barra_return = stock_barra_return[stock_barra_return['trade_date'] == end_date]
    barra_cols = ['beta', 'momentum', 'size', 'earnyild',
                  'resvol', 'growth', 'btop', 'leverage', 'liquidty', 'sizenl',
                  'agriculture', 'automobiles', 'banks', 'buildmater', 'computers',
                  'conglomerates', 'constrdecor', 'defense', 'electronics',
                  'foodbeverages', 'healthcare', 'homeappliances', 'lightindustry',
                  'machineequip', 'media', 'nonbankfinan', 'nonferrousmetals',
                  'realestate', 'steel', 'telecoms', 'transportation', 'utilities',
                  'basicchemicals', 'beautycare', 'coal', 'environprotect', 'petroleum',
                  'powerequip', 'retailtrade', 'socialservices', 'textileapparel',
                  'country']
    return_df = pd.DataFrame(columns=barra_cols + ['spret'])
    # 计算指数暴露
    index_weight = index_weight[['security_code', 'weight']]
    index_weight = pd.merge(left=index_weight, right=barra_all)
    index_weight = pd.merge(left=index_weight, right=stock_barra_return[['security_code', 'spret']])
    return_df.loc['index_weight'] = np.dot(index_weight['weight'], index_weight[barra_cols + ['spret']])
    # 计算持仓暴露
    total_asset = past_holding['NetHoldingValue'].sum()
    past_holding = past_holding[['Windcode', 'Position']]
    past_holding.columns = ['security_code', 'position']
    past_holding['security_code'] = past_holding.apply(lambda x: x['security_code'][:6], axis=1)
    past_holding = pd.merge(left=past_holding[['security_code', 'position']],
                            right=stock_df_today[['security_code', 's_dq_close']], how='inner')
    past_holding['weight'] = past_holding['position'] * past_holding['s_dq_close'] / total_asset
    past_holding = past_holding[past_holding['weight'] != 0]
    past_holding = past_holding[['security_code', 'weight']]
    past_holding = pd.merge(left=past_holding, right=barra_all)
    past_holding = pd.merge(left=past_holding, right=stock_barra_return[['security_code', 'spret']])
    return_df.loc['fund_weight'] = np.dot(past_holding['weight'], past_holding[barra_cols + ['spret']])
    # 计算收益
    return_df.loc['barra_return', barra_cols] = barra_return[barra_cols].iloc[0, :]
    return_df.loc['barra_return', 'spret'] = 0.01  # 股票特异性收益已经计算完了,0.01为去百分比
    return_df.loc['exposure'] = return_df.loc['fund_weight'] - return_df.loc['index_weight']
    return_df.loc['return'] = return_df.loc['exposure'] * return_df.loc['barra_return'] * 100
    return_df.loc['return'].sum()
    return_df.loc['sum'] = np.nan
    return_df.loc['sum', 'beta'] = '风格暴露收益'
    return_df.loc['sum', 'momentum'] = return_df.loc['return', 'beta':'sizenl'].sum()
    return_df.loc['sum', 'size'] = '行业暴露收益'
    return_df.loc['sum', 'earnyild'] = return_df.loc['return', 'agriculture':'country'].sum()
    return_df.loc['sum', 'resvol'] = '个股特异性收益'
    return_df.loc['sum', 'growth'] = return_df.loc['return', 'spret']
    return_df.loc['sum', 'btop'] = '总收益'
    return_df.loc['sum', 'leverage'] = return_df.loc['return'].sum()
    return_df.reset_index(inplace=True)

    # # 进行因子归因
    # model_training_date = get_adjusted_date(trade_date_list, frequency=mc.adjustment_frequency)
    # model_date = [date_ for date_ in model_training_date if date_ <= pd.to_datetime(end_date)][-1].strftime('%Y%m%d')
    # factor_df = pd.DataFrame(columns=['factor'])
    # weight_list = []
    # for T in mc.T_pre_list:
    #     weight_list.append('weight_%s' % T)
    #     xgb_model = pickle.load(open(os.path.join(jc.model_path, 'xgb_model_%s_%s.pkl' % (T, model_date)), "rb"))
    #     factor_df_ = pd.DataFrame(
    #         {'factor': xgb_model.get_booster().feature_names, 'weight_%s' % T: xgb_model.feature_importances_})
    #     # 个股
    #     factor_df = pd.merge(left=factor_df, right=factor_df_, on='factor', how='outer')
    # factor_df.fillna(0, inplace=True)
    # factor_df['weight'] = factor_df[weight_list].sum(axis=1)
    # stock_pool = pd.read_pickle('Database/Origin_Database/stock_pool.pkl')
    #
    #
    #
    #
    #
    # stock_pool = stock_pool[stock_pool['trade_date'] == end_date]
    # stock_pool = pd.merge(left=stock_pool, right=stock_df_today[['security_code', 'trade_date', 'chg']],
    #                       on=['security_code', 'trade_date'], how='left')
    # for factor_ in factor_df['factor']:
    #     factor_data = pd.read_pickle('Database/Neut_Database/%s_neut.pkl'%factor_)
    #
    # a = pd.read_pickle('Database/Neut_Database/ROC_20_neut.pkl')
    # a = a[a['security_code'] == '000670']

    with pd.ExcelWriter('daily_result/%s日收益归因.xlsx' % end_date) as writer:
        return_df.to_excel(writer, sheet_name='风格归因', index=False)


def daily_trade_future(end_date):
    score_df = pd.read_pickle('daily_result/%s日股票预测得分.pkl' % end_date)
    col_list = []  # 打分基础
    for num in mc.T_pre_list:
        for model in mc.model_list:
            col_list.append(model + '_' + str(num))

    for model in col_list:
        score_df_tmp = score_df[['security_code', 'trade_date', model]]
        score_df_tmp['%s_mean' % model] = score_df.groupby('trade_date')[model].transform('mean')
        score_df_tmp['%s_std' % model] = score_df.groupby('trade_date')[model].transform('std')
        score_df[model] = (score_df_tmp[model] - score_df_tmp['%s_mean' % model]) / score_df_tmp['%s_std' % model]
    score_df['score_final'] = score_df.iloc[:, -len(col_list):].mean(axis=1)
    score_df = score_df[['security_code', 'trade_date', 'score_final']]
    # 读取股票数据
    stock_df = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_df = stock_df[stock_df['trade_date'] == end_date]  # 只取当日数据
    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_df[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.dropna(inplace=True)  # 只保留池内数据
    pool_list = list(score_df['security_code'])
    # score_df.fillna(-9999, inplace=True)  #  不在选股池内，填负值

    # 指数数据整理
    # 生成指数成分股的barra和行业
    chosen_index = jc.benchmark
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    # df_index = df_index[df_index['trade_date'] == end_date]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 取权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    index_weight = index_weight[index_weight['trade_date'] == end_date]

    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)

    index_weight = index_weight[index_weight['weight'] != 0]
    index_weight[barra_list] = index_weight[barra_list].T.mul(list(index_weight['weight'])).T * len(index_weight)
    df_index.loc[end_date, barra_list] = index_weight[barra_list].mean()  # 指数barra
    df_index.loc[end_date, barra_std_list] = list(index_weight[barra_list].std())  # 指数barra标准差
    df_index.loc[end_date, industry_list] = index_weight['sw_industry'].value_counts() / len(index_weight)
    df_index.fillna(0, inplace=True)
    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 读取昨日持仓
    past_holding = w.wpf("1000多周期拟合模拟盘", "NetHoldingValue,Position",
                         "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1" % end_date.replace('-', ''),
                         usedf=True)[1]
    # past_holding = w.wpf("1000指增0.5标准差", "NetHoldingValue,Position",
    #                      "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1"%end_date.replace('-', ''),
    #                      usedf=True)[1]
    if past_holding.shape[1] != 5:  # 初始日期
        past_holding = pd.DataFrame({'AssetClass': ['现金'], 'Windcode': ['CNY'],
                                     'AssetName': ['人民币'], 'NetHoldingValue': [10000000], 'Position': [10000000]})
    else:
        past_holding.sort_values(by='Windcode', inplace=True)
    # try:
    #     cal_factor_attribution(past_holding, index_weight, end_date)  # 首日无法归因
    # except:
    #     pass
    total_asset = past_holding['NetHoldingValue'].sum()
    past_weight = past_holding[past_holding['AssetClass'] != '现金']
    past_weight['past_weight'] = past_weight['NetHoldingValue'] / total_asset
    past_weight = past_weight[['Windcode', 'past_weight', 'Position']]
    past_weight.columns = ['security_code', 'past_weight', 'position']
    if len(past_weight) != 0:
        past_weight['security_code'] = past_weight.apply(lambda x: x['security_code'][:6], axis=1)
    past_weight = pd.merge(left=stock_df[['security_code']], right=past_weight,
                           on='security_code', how='outer')
    past_weight.fillna(0, inplace=True)
    total_list = list(past_weight['security_code'])
    remove_list = [code_ for code_ in total_list if code_ not in pool_list]
    # 调整顺序
    past_weight = past_weight.sort_values(by='security_code',
                                          key=lambda x: x.map({v: i for i, v in enumerate(pool_list + remove_list)}))

    # 计算当日opt_weight, 无解情况下则放宽限制
    barra_std_num = mc.barra_std_num
    # past_weight.sort_values(by='security_code', inplace=True)
    # score_df.sort_values(by='security_code', inplace=True)
    try:
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=0.15)
    except:
        print('1000优化失败，需要提高换手率')
        try:
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
        except:
            print('1000换手提高后仍优化失败，需要放大barra敞口')
            barra_std_num += mc.barra_std_num / 2
            for barra_ in barra_list:
                df_index.loc[end_date, barra_ + '_lower'] = (df_index[barra_] - barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
                df_index.loc[end_date, barra_ + '_upper'] = (df_index[barra_] + barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
    turnover = 0.2
    while status not in ['optimal', 'optimal_inaccurate']:
        turnover += 0.05
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=turnover)

    target_df = pd.DataFrame(opt_weight, index=[0]).T.reset_index()
    target_df.columns = ['security_code', 'target_weight']
    target_df['target_value'] = target_df['target_weight'] * total_asset
    target_df = pd.merge(left=target_df, right=stock_df[['security_code', 's_dq_close']],
                         on=['security_code'], how='left')
    target_df['target_num'] = np.floor(target_df['target_value'] / target_df['s_dq_close'] / 100) * 100
    weight_diff = pd.merge(left=past_weight, right=target_df, on='security_code', how='outer')
    weight_diff.fillna(0, inplace=True)
    weight_diff = weight_diff[(weight_diff['past_weight'] != 0) | (weight_diff['target_weight'] != 0)]  # 有交易的
    weight_diff['diff'] = weight_diff['target_weight'] - weight_diff['past_weight']
    weight_diff.loc[abs(weight_diff['diff']) < 0.001, 'diff'] = 0  # 差距过小的不进行交易

    # 获取下一个交易日日期
    trade_date = pd.to_datetime(w.tdaysoffset(1, end_date, "").Data[0][0]).strftime('%Y-%m-%d')
    weight_diff['security_code'] = weight_diff.apply(lambda x: x['security_code'] + '.SH'
    if x['security_code'][0] == '6' else x['security_code'] + '.SZ', axis=1)
    total_order = pd.DataFrame()
    # 买单
    buy_order = weight_diff[weight_diff['diff'] > 0]
    buy_order['buy_num'] = buy_order['target_num'] - buy_order['position']
    buy_order['open'] = np.nan
    buy_order = buy_order[['security_code', 'buy_num', 'open']]
    buy_order['trade_date'] = trade_date
    buy_order['operate'] = '买入'
    buy_order = buy_order[['trade_date', 'security_code', 'buy_num', 'open', 'operate']]
    buy_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
    buy_order = buy_order[buy_order['买卖数量'] > 0]
    total_order = pd.concat([total_order, buy_order], axis=0)
    # 卖单
    sell_order = weight_diff[((weight_diff['diff'] < 0) & (weight_diff['target_num'] != 0))
                             | (weight_diff['target_num'] == 0)]  # 要大幅减仓的 以及 要清仓的
    if len(sell_order) != 0:  # 有要卖出的
        sell_order['open'] = np.nan
        sell_order['sell_num'] = sell_order['position'] - sell_order['target_num']
        sell_order = sell_order[['security_code', 'sell_num', 'open']]
        sell_order['trade_date'] = trade_date
        sell_order['operate'] = '卖出'
        sell_order = sell_order[['trade_date', 'security_code', 'sell_num', 'open', 'operate']]
        sell_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
        sell_order = sell_order[sell_order['买卖数量'] > 0]
        total_order = pd.concat([total_order, sell_order], axis=0)

    total_order.to_excel('trade/1000指增交易单%s.xlsx' % trade_date, index=False)


def shipan_trade(end_date):
    score_df = pd.read_pickle('daily_result/%s日股票预测得分.pkl' % end_date)
    col_list = []  # 打分基础
    for num in mc.T_pre_list:
        for model in mc.model_list:
            col_list.append(model + '_' + str(num))

    for model in col_list:
        score_df_tmp = score_df[['security_code', 'trade_date', model]]
        score_df_tmp['%s_mean' % model] = score_df.groupby('trade_date')[model].transform('mean')
        score_df_tmp['%s_std' % model] = score_df.groupby('trade_date')[model].transform('std')
        score_df[model] = (score_df_tmp[model] - score_df_tmp['%s_mean' % model]) / score_df_tmp['%s_std' % model]
    score_df['score_final'] = score_df.iloc[:, -len(col_list):].mean(axis=1)
    score_df = score_df[['security_code', 'trade_date', 'score_final']]
    # 剔除限制性清单的个股
    ban_df = pd.read_excel('限制清单0624.xlsx')
    ban_df.dropna(subset='证券代码', inplace=True)
    ban_df['证券代码'] = ban_df.apply(lambda x: str(int(x['证券代码'])).zfill(6), axis=1)
    ban_list = list(ban_df['证券代码'].unique())
    score_df = score_df[~score_df['security_code'].isin(ban_list)]

    # 读取股票数据
    stock_df = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_df = stock_df[stock_df['trade_date'] == end_date]  # 只取当日数据
    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_df[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.dropna(inplace=True)  # 只保留池内数据
    pool_list = list(score_df['security_code'])
    # score_df.fillna(-9999, inplace=True)  #  不在选股池内，填负值

    # 指数数据整理
    # 生成指数成分股的barra和行业
    chosen_index = jc.benchmark
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    # df_index = df_index[df_index['trade_date'] == end_date]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 取权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    index_weight = index_weight[index_weight['trade_date'] == end_date]

    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)

    index_weight = index_weight[index_weight['weight'] != 0]
    index_weight[barra_list] = index_weight[barra_list].T.mul(list(index_weight['weight'])).T * len(index_weight)
    df_index.loc[end_date, barra_list] = index_weight[barra_list].mean()  # 指数barra
    df_index.loc[end_date, barra_std_list] = list(index_weight[barra_list].std())  # 指数barra标准差
    df_index.loc[end_date, industry_list] = index_weight['sw_industry'].value_counts() / len(index_weight)
    df_index.fillna(0, inplace=True)
    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 读取昨日持仓
    past_holding = w.wpf("1000多周期拟合实盘", "NetHoldingValue,Position",
                         "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1" % end_date.replace('-', ''),
                         usedf=True)[1]
    # past_holding = w.wpf("1000指增0.5标准差", "NetHoldingValue,Position",
    #                      "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1"%end_date.replace('-', ''),
    #                      usedf=True)[1]
    if past_holding.shape[1] != 5:  # 初始日期
        past_holding = pd.DataFrame({'AssetClass': ['现金'], 'Windcode': ['CNY'],
                                     'AssetName': ['人民币'], 'NetHoldingValue': [5000000], 'Position': [5000000]})
    else:
        past_holding.sort_values(by='Windcode', inplace=True)
    # try:
    #     cal_factor_attribution(past_holding, index_weight, end_date)  # 首日无法归因
    # except:
    #     pass
    total_asset = past_holding['NetHoldingValue'].sum()
    past_weight = past_holding[past_holding['AssetClass'] != '现金']
    past_weight['past_weight'] = past_weight['NetHoldingValue'] / total_asset
    past_weight = past_weight[['Windcode', 'past_weight', 'Position']]
    past_weight.columns = ['security_code', 'past_weight', 'position']
    if len(past_weight) != 0:
        past_weight['security_code'] = past_weight.apply(lambda x: x['security_code'][:6], axis=1)
    past_weight = pd.merge(left=stock_df[['security_code']], right=past_weight,
                           on='security_code', how='outer')
    past_weight.fillna(0, inplace=True)
    total_list = list(past_weight['security_code'])
    remove_list = [code_ for code_ in total_list if code_ not in pool_list]
    # 调整顺序
    past_weight = past_weight.sort_values(by='security_code',
                                          key=lambda x: x.map({v: i for i, v in enumerate(pool_list + remove_list)}))

    # 计算当日opt_weight, 无解情况下则放宽限制
    barra_std_num = mc.barra_std_num
    # past_weight.sort_values(by='security_code', inplace=True)
    # score_df.sort_values(by='security_code', inplace=True)
    try:
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=0.15)
    except:
        print('1000优化失败，需要提高换手率')
        try:
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
        except:
            print('1000换手提高后仍优化失败，需要放大barra敞口')
            barra_std_num += mc.barra_std_num / 2
            for barra_ in barra_list:
                df_index.loc[end_date, barra_ + '_lower'] = (df_index[barra_] - barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
                df_index.loc[end_date, barra_ + '_upper'] = (df_index[barra_] + barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
    turnover = 0.2
    while status not in ['optimal', 'optimal_inaccurate']:
        turnover += 0.05
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=turnover)

    target_df = pd.DataFrame(opt_weight, index=[0]).T.reset_index()
    target_df.columns = ['security_code', 'target_weight']
    target_df['target_value'] = target_df['target_weight'] * total_asset
    target_df = pd.merge(left=target_df, right=stock_df[['security_code', 's_dq_close']],
                         on=['security_code'], how='left')
    target_df['target_num'] = np.floor(target_df['target_value'] / target_df['s_dq_close'] / 100) * 100
    weight_diff = pd.merge(left=past_weight, right=target_df, on='security_code', how='outer')
    weight_diff.fillna(0, inplace=True)
    weight_diff = weight_diff[(weight_diff['past_weight'] != 0) | (weight_diff['target_weight'] != 0)]  # 有交易的
    weight_diff['diff'] = weight_diff['target_weight'] - weight_diff['past_weight']
    weight_diff.loc[abs(weight_diff['diff']) < 0.001, 'diff'] = 0  # 差距过小的不进行交易

    # 获取下一个交易日日期
    trade_date = pd.to_datetime(w.tdaysoffset(1, end_date, "").Data[0][0]).strftime('%Y-%m-%d')
    weight_diff['security_code'] = weight_diff.apply(lambda x: x['security_code'] + '.SH'
    if x['security_code'][0] == '6' else x['security_code'] + '.SZ', axis=1)
    total_order = pd.DataFrame()
    # 买单
    buy_order = weight_diff[weight_diff['diff'] > 0]
    buy_order['buy_num'] = buy_order['target_num'] - buy_order['position']
    buy_order['open'] = np.nan
    buy_order = buy_order[['security_code', 'buy_num', 'open']]
    buy_order['trade_date'] = trade_date
    buy_order['operate'] = '买入'
    buy_order = buy_order[['trade_date', 'security_code', 'buy_num', 'open', 'operate']]
    buy_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
    buy_order = buy_order[buy_order['买卖数量'] > 0]
    total_order = pd.concat([total_order, buy_order], axis=0)
    # 卖单
    sell_order = weight_diff[((weight_diff['diff'] < 0) & (weight_diff['target_num'] != 0))
                             | (weight_diff['target_num'] == 0)]  # 要大幅减仓的 以及 要清仓的
    if len(sell_order) != 0:  # 有要卖出的
        sell_order['open'] = np.nan
        sell_order['sell_num'] = sell_order['position'] - sell_order['target_num']
        sell_order = sell_order[['security_code', 'sell_num', 'open']]
        sell_order['trade_date'] = trade_date
        sell_order['operate'] = '卖出'
        sell_order = sell_order[['trade_date', 'security_code', 'sell_num', 'open', 'operate']]
        sell_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
        sell_order = sell_order[sell_order['买卖数量'] > 0]
        total_order = pd.concat([total_order, sell_order], axis=0)

    total_order.to_excel('trade/实盘_1000指增交易单%s.xlsx' % trade_date, index=False)


def daily_trade_future_300(end_date):
    score_df = pd.read_pickle('daily_result/%s日股票预测得分.pkl' % end_date)
    col_list = []  # 打分基础
    for num in mc.T_pre_list:
        for model in mc.model_list:
            col_list.append(model + '_' + str(num))

    for model in col_list:
        score_df_tmp = score_df[['security_code', 'trade_date', model]]
        score_df_tmp['%s_mean' % model] = score_df.groupby('trade_date')[model].transform('mean')
        score_df_tmp['%s_std' % model] = score_df.groupby('trade_date')[model].transform('std')
        score_df[model] = (score_df_tmp[model] - score_df_tmp['%s_mean' % model]) / score_df_tmp['%s_std' % model]
    score_df['score_final'] = score_df.iloc[:, -len(col_list):].mean(axis=1)
    score_df = score_df[['security_code', 'trade_date', 'score_final']]
    # 读取股票数据
    stock_df = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_df = stock_df[stock_df['trade_date'] == end_date]  # 只取当日数据
    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_df[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.dropna(inplace=True)  # 只保留池内数据
    pool_list = list(score_df['security_code'])
    # score_df.fillna(-9999, inplace=True)  #  不在选股池内，填负值

    # 指数数据整理
    # 生成指数成分股的barra和行业
    chosen_index = jc.benchmark
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    # df_index = df_index[df_index['trade_date'] == end_date]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 取权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    index_weight = index_weight[index_weight['trade_date'] == end_date]

    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)

    index_weight = index_weight[index_weight['weight'] != 0]
    index_weight[barra_list] = index_weight[barra_list].T.mul(list(index_weight['weight'])).T * len(index_weight)
    df_index.loc[end_date, barra_list] = index_weight[barra_list].mean()  # 指数barra
    df_index.loc[end_date, barra_std_list] = list(index_weight[barra_list].std())  # 指数barra标准差
    df_index.loc[end_date, industry_list] = index_weight['sw_industry'].value_counts() / len(index_weight)
    df_index.fillna(0, inplace=True)
    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 读取昨日持仓
    past_holding = w.wpf("300多周期拟合模拟盘", "NetHoldingValue,Position",
                         "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1" % end_date.replace('-', ''),
                         usedf=True)[1]
    # past_holding = w.wpf("1000指增0.5标准差", "NetHoldingValue,Position",
    #                      "view=PMS;date=%s;Currency=BSY;sectorcode=101;displaymode=1"%end_date.replace('-', ''),
    #                      usedf=True)[1]
    if past_holding.shape[1] != 5:  # 初始日期
        past_holding = pd.DataFrame({'AssetClass': ['现金'], 'Windcode': ['CNY'],
                                     'AssetName': ['人民币'], 'NetHoldingValue': [10000000], 'Position': [10000000]})
    else:
        past_holding.sort_values(by='Windcode', inplace=True)
    # try:
    #     cal_factor_attribution(past_holding, index_weight, end_date)  # 首日无法归因
    # except:
    #     pass
    total_asset = past_holding['NetHoldingValue'].sum()
    past_weight = past_holding[past_holding['AssetClass'] != '现金']
    past_weight['past_weight'] = past_weight['NetHoldingValue'] / total_asset
    past_weight = past_weight[['Windcode', 'past_weight', 'Position']]
    past_weight.columns = ['security_code', 'past_weight', 'position']
    if len(past_weight) != 0:
        past_weight['security_code'] = past_weight.apply(lambda x: x['security_code'][:6], axis=1)
    past_weight = pd.merge(left=stock_df[['security_code']], right=past_weight,
                           on='security_code', how='outer')
    past_weight.fillna(0, inplace=True)
    total_list = list(past_weight['security_code'])
    remove_list = [code_ for code_ in total_list if code_ not in pool_list]
    # 调整顺序
    past_weight = past_weight.sort_values(by='security_code',
                                          key=lambda x: x.map({v: i for i, v in enumerate(pool_list + remove_list)}))

    # 计算当日opt_weight, 无解情况下则放宽限制
    barra_std_num = mc.barra_std_num
    # past_weight.sort_values(by='security_code', inplace=True)
    # score_df.sort_values(by='security_code', inplace=True)
    try:
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=0.15)
    except:
        print('300优化失败，需要提高换手率')
        try:
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
        except:
            print('300换手提高后仍优化失败，需要放大barra敞口')
            barra_std_num += mc.barra_std_num / 2
            for barra_ in barra_list:
                df_index.loc[end_date, barra_ + '_lower'] = (df_index[barra_] - barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
                df_index.loc[end_date, barra_ + '_upper'] = (df_index[barra_] + barra_std_num *
                                                             df_index[barra_ + '_std']).loc[end_date]
            opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                          score_df, df_index, mc.turnover_punish, turnover=0.2)
    turnover = 0.2
    while status not in ['optimal', 'optimal_inaccurate']:
        turnover += 0.05
        opt_weight, status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                      score_df, df_index, mc.turnover_punish, turnover=turnover)

    target_df = pd.DataFrame(opt_weight, index=[0]).T.reset_index()
    target_df.columns = ['security_code', 'target_weight']
    target_df['target_value'] = target_df['target_weight'] * total_asset
    target_df = pd.merge(left=target_df, right=stock_df[['security_code', 's_dq_close']],
                         on=['security_code'], how='left')
    target_df['target_num'] = np.floor(target_df['target_value'] / target_df['s_dq_close'] / 100) * 100
    weight_diff = pd.merge(left=past_weight, right=target_df, on='security_code', how='outer')
    weight_diff.fillna(0, inplace=True)
    weight_diff = weight_diff[(weight_diff['past_weight'] != 0) | (weight_diff['target_weight'] != 0)]  # 有交易的
    weight_diff['diff'] = weight_diff['target_weight'] - weight_diff['past_weight']
    weight_diff.loc[abs(weight_diff['diff']) < 0.001, 'diff'] = 0  # 差距过小的不进行交易

    # 获取下一个交易日日期
    trade_date = pd.to_datetime(w.tdaysoffset(1, end_date, "").Data[0][0]).strftime('%Y-%m-%d')
    weight_diff['security_code'] = weight_diff.apply(lambda x: x['security_code'] + '.SH'
    if x['security_code'][0] == '6' else x['security_code'] + '.SZ', axis=1)
    total_order = pd.DataFrame()
    # 买单
    buy_order = weight_diff[weight_diff['diff'] > 0]
    buy_order['buy_num'] = buy_order['target_num'] - buy_order['position']
    buy_order['open'] = np.nan
    buy_order = buy_order[['security_code', 'buy_num', 'open']]
    buy_order['trade_date'] = trade_date
    buy_order['operate'] = '买入'
    buy_order = buy_order[['trade_date', 'security_code', 'buy_num', 'open', 'operate']]
    buy_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
    buy_order = buy_order[buy_order['买卖数量'] > 0]
    total_order = pd.concat([total_order, buy_order], axis=0)
    # 卖单
    sell_order = weight_diff[((weight_diff['diff'] < 0) & (weight_diff['target_num'] != 0))
                             | (weight_diff['target_num'] == 0)]  # 要大幅减仓的 以及 要清仓的
    if len(sell_order) != 0:  # 有要卖出的
        sell_order['open'] = np.nan
        sell_order['sell_num'] = sell_order['position'] - sell_order['target_num']
        sell_order = sell_order[['security_code', 'sell_num', 'open']]
        sell_order['trade_date'] = trade_date
        sell_order['operate'] = '卖出'
        sell_order = sell_order[['trade_date', 'security_code', 'sell_num', 'open', 'operate']]
        sell_order.columns = ['买卖日期', '证券代码', '买卖数量', '买卖价格', '买卖方向']
        sell_order = sell_order[sell_order['买卖数量'] > 0]
        total_order = pd.concat([total_order, sell_order], axis=0)

    total_order.to_excel('trade/300指增交易单%s.xlsx' % trade_date, index=False)


def chosen_factor_performance(model_date, end_date, T):
    from datetime import timedelta
    model_date = '20240401'
    end_date = '2024-04-19'
    T = 5

    # 读取股票数据
    stock_df = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))  # 导入stock价格文件
    # trade_date_list = [pd.to_datetime(date_) for date_ in stock_df['trade_date'].unique()
    #                  if date_ < pd.to_datetime(end_date)]  # 避免未来数据，要扣除最后几天的数据
    train_period = mc.train_window_dict[str(T)]
    # end_date = trade_date_list[-T]
    start_date = pd.to_datetime(end_date) - timedelta(days=train_period)
    stock_df = stock_df[(stock_df['trade_date'] >= start_date) & (stock_df['trade_date'] <= end_date)]
    stock_df['open'] = stock_df['s_dq_open'] * stock_df['s_dq_adjclose'] / stock_df['s_dq_close']  # 后复权后开盘价
    stock_df['next_open'] = stock_df.groupby('security_code')['open'].shift(-1)
    stock_df['next_open'] = stock_df['next_open'].fillna(stock_df['s_dq_adjclose'])

    # stock_price只取日期、股票代码，作为表头
    factor_df = pd.read_pickle(os.path.join(jc.database_path, 'stock_pool.pkl'))
    stock_df = pd.merge(left=factor_df, right=stock_df)

    # 合并行业信息
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))  # 获取个股和行业对应关系
    industry_cate.drop_duplicates(inplace=True)
    stock_df = pd.merge(stock_df, industry_cate, on=['security_code', 'trade_date'], how='outer')
    del industry_cate  # 省内存
    stock_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    stock_df['sw_industry'] = stock_df.groupby('security_code')['sw_industry'].ffill()
    stock_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据

    price_df = stock_df.copy()
    price_df['target_open'] = price_df.groupby('security_code')['next_open'].shift(-1)
    price_df.dropna(inplace=True)
    price_df['stock_yield'] = price_df['target_open'] / price_df['next_open'] - 1
    industry_price = price_df.groupby(['trade_date', 'sw_industry'])['stock_yield'].mean().reset_index()
    industry_price.rename(columns={'stock_yield': 'index_yield'}, inplace=True)
    price_df = pd.merge(left=price_df, right=industry_price, on=['trade_date', 'sw_industry'], how='left')
    price_df['yield'] = price_df['stock_yield'] - price_df['index_yield']
    price_df = price_df[['security_code', 'trade_date', 'yield']]

    model_date = '20240301'
    xgb_model = pickle.load(open(os.path.join(jc.model_path, 'xgb_model_%s_%s.pkl' % (T, model_date)), "rb"))
    factor_importance = pd.DataFrame(
        {'factor': xgb_model.get_booster().feature_names, 'weight_%s' % T: xgb_model.feature_importances_})
    xgb_model = pickle.load(open(os.path.join(jc.model_path, 'xgb_model_%s_%spc.pkl' % (T, model_date)), "rb"))
    factor_importancepc = pd.DataFrame(
        {'factor': xgb_model.get_booster().feature_names, 'weight_%s' % T: xgb_model.feature_importances_})

    # # 单进程计算多因子
    # for factor in fc.factor_list:
    #     df_input = [factor, price_df, T]
    #     df_out = multi_factor_performance(df_input)
    #     result_df.loc[len(result_df)] = df_out[0]
    #     if df_out[4] == 1:
    #         l_s_dict[df_out[0][0]] = df_out[2]
    #         l_dict[df_out[0][0]] = df_out[3]
    #         diff_curve_dict[df_out[0][0]] = df_out[1]


# if __name__ == '__main__':
#     # 回测版本
#     # backtest_job()

#     # 实盘数据更新
#     today = dt.date.today().strftime('%Y-%m-%d')
#     # today = '2024-06-21'

#     print('开始更新%s日程序' % today)
#     end_date = pd.to_datetime(w.tdaysoffset(-1, today, "").Data[0][0]).strftime('%Y-%m-%d')
#     daily_job(end_date)

#     # 1000指增: 换手惩罚100，barra敞口0.5
#     jc.benchmark = '000852'
#     mc.turnover_punish = 100
#     mc.barra_std_num = 0.5
#     mc.T_pre_list = [5, 10, 22]
#     daily_trade_future(end_date)  # 行业敞口2%，无市值绝对限制，风险敞口0.5std

#     # 实盘1000指增：换手惩罚100，barra敞口0.5
#     jc.benchmark = '000852'
#     mc.turnover_punish = 100
#     mc.barra_std_num = 0.5
#     mc.T_pre_list = [5, 10, 22]
#     shipan_trade(end_date)

#     # 300指增：换手惩罚300，barra敞口0.3
#     jc.benchmark = '000300'
#     mc.turnover_punish = 300
#     mc.barra_std_num = 0.3
#     mc.T_pre_list = [10, 22]
#     daily_trade_future_300(end_date)  # 行业敞口2%，无市值绝对限制，风险敞口0.5std


# test the function
if __name__ == '__main__':
    backtest_job_multi()


