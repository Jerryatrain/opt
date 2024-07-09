import os
import time

import numpy
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import timedelta
import job_config as jc
import model_config as mc
# from Functions.doc_functions import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial Unicode MS']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
import matplotlib
matplotlib.use('Agg')  # 不显示图片


def test_add_table(d, table): # 为输出docx做准备
  t = d.add_table(table.shape[0]+1, table.shape[1])
  for j in range(table.shape[-1]):
    t.cell(0,j).text = table.columns[j]
  for i in range(table.shape[0]):
    for j in range(table.shape[-1]):
        t.cell(i+1,j).text = str(table.values[i,j])


def port_opt(period_list, industry_list, barra_list, past_weight, stock_today, index_today, turnover_punish, turnover=0.15):
    """
        求某日的最优投资组合
    """

    # 定义优化变量
    weights = cp.Variable(shape=(len(stock_today), 1), nonneg=True)  # 指定下限,默认下限为0

    # 约束集合
    constraints = []

    # 约束1：控制个股权重之和为1
    constraints.append(cp.sum(weights) == 1)

    # 约束2：个股权重小于1%
    constraints.append(weights <= 0.01)

    # 约束3： 控制行业偏离不超过2%
    # 将DataFrame的某一列改为哑变量
    industry_dummies = pd.get_dummies(stock_today['sw_industry'], prefix='industry')
    stock_today_industry = pd.concat([stock_today, industry_dummies], axis=1)
    industry_list_name = ['industry_' + industry for industry in industry_list]
    try:
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                      :] <= mc.industry_diff)
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                      :] >= -mc.industry_diff)
    except:
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list] <= mc.industry_diff)
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list] >= -mc.industry_diff)

    # 约束4：Barra偏离不超过0.5个标准差
    barra_upper_name = [barra_ + '_upper' for barra_ in barra_list]
    barra_lower_name = [barra_ + '_lower' for barra_ in barra_list]
    try:
        constraints.append(weights[:, 0] @ stock_today[barra_list] >= index_today[barra_lower_name].iloc[0, :])
        constraints.append(weights[:, 0] @ stock_today[barra_list] <= index_today[barra_upper_name].iloc[0, :])
    except:
        constraints.append(weights[:, 0] @ stock_today[barra_list] >= index_today[barra_lower_name])
        constraints.append(weights[:, 0] @ stock_today[barra_list] <= index_today[barra_upper_name])
    # 计算交易权重变化
    try:
        trade_cost_sum = (cp.sum(cp.abs(weights - past_weight[['past_weight']].iloc[:len(stock_today)]))
                          + cp.sum(past_weight[['past_weight']].iloc[len(stock_today):]))
    except:
        trade_cost_sum = cp.sum(cp.abs(weights - past_weight[['past_weight']].iloc[:len(stock_today)]))
    # 约束5：限制换手
    if past_weight['past_weight'].sum() > 0.5:  # 首日排除
        constraints.append(trade_cost_sum <= turnover * 2)  # 单日换手不超过15%

    # 优化目标函数
    profit_sum = cp.sum(cp.multiply(weights, stock_today[period_list]))
    obj = profit_sum - (mc.trade_cost + (mc.buy_fee + mc.sell_fee) / 2) * trade_cost_sum * turnover_punish
    # 定义问题
    problem = cp.Problem(cp.Maximize(obj), constraints)
    # 求解
    try:
        problem.solve(solver='ECOS', qcp=True, max_iters=2000)
    except:
        problem.solve()
    # problem.solve(solver='ECOS', qcp=True, max_iters=1000, abstol=10 ** (-8), reltol=10 ** (-8), feastol=10 ** (-8),
    #               abstol_inacc=5 * 10 ** (-5), reltol_inacc=5 * 10 ** (-5), feastol_inacc=10 ** (-4))
    # 获取解的值并返回
    weights_value = weights.value
    weights_value = pd.concat([stock_today[['security_code']].reset_index(drop=True),
                               pd.DataFrame(weights_value[:, 0], columns=['weight']).reset_index(drop=True)],
                              axis=1)
    weights_value.loc[weights_value['weight'] < 0.001, 'weight'] = 0
    weights_value = weights_value[weights_value['weight'] != 0]
    weights_value['weight'] = weights_value['weight'] / weights_value['weight'].sum()
    weights_value = pd.merge(left=weights_value, right=past_weight[['security_code']], on='security_code', how='outer')
    weights_value.fillna(0, inplace=True)
    weights_value = weights_value.set_index('security_code')['weight'].to_dict()
    return weights_value, problem.status


def port_opt_cov(period_list, industry_list, barra_list, past_weight, stock_today, index_today, turnover_punish,
                 cov_matrix, turnover=0.15):
    """
        求某日的最优投资组合
    """

    # 定义优化变量
    weights = cp.Variable(shape=(len(past_weight), 1), nonneg=True)  # 指定下限,默认下限为0

    # 约束集合
    constraints = []

    # 约束1：控制个股权重之和为1
    constraints.append(cp.sum(weights) == 1)

    # 约束2：个股权重小于1%
    constraints.append(weights <= 0.01)

    # 约束3： 控制行业偏离不超过2%
    # 将DataFrame的某一列改为哑变量
    industry_dummies = pd.get_dummies(stock_today['sw_industry'], prefix='industry')
    stock_today_industry = pd.concat([stock_today, industry_dummies], axis=1)
    industry_list_name = ['industry_' + industry for industry in industry_list]
    try:
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                      :] <= mc.industry_diff)
        constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                      :] >= -mc.industry_diff)
    except:
        constraints.append(
            weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list] <= mc.industry_diff)
        constraints.append(
            weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list] >= -mc.industry_diff)

    # 约束4：Barra偏离不超过0.3个标准差
    barra_upper_name = [barra_ + '_upper' for barra_ in barra_list]
    barra_lower_name = [barra_ + '_lower' for barra_ in barra_list]
    try:
        constraints.append(weights[:, 0] @ stock_today[barra_list] >= index_today[barra_lower_name].iloc[0, :])
        constraints.append(weights[:, 0] @ stock_today[barra_list] <= index_today[barra_upper_name].iloc[0, :])
    except:
        constraints.append(weights[:, 0] @ stock_today[barra_list] >= index_today[barra_lower_name])
        constraints.append(weights[:, 0] @ stock_today[barra_list] <= index_today[barra_upper_name])

    # 计算交易权重变化
    trade_cost_sum = cp.sum(cp.abs(weights - past_weight[['past_weight']]))

    # 约束5：限制换手
    if past_weight['past_weight'].sum() > 0.5:  # 首日排除
        constraints.append(trade_cost_sum <= turnover * 2)  # 单日换手不超过15%

    # 优化目标函数
    profit_sum = cp.sum(cp.multiply(weights, stock_today[period_list]))
    risk_aversion = mc.risk_aversion  # 风险厌恶系数
    trade_penalty = (mc.trade_cost + (mc.buy_fee + mc.sell_fee) / 2) * trade_cost_sum * turnover_punish
    risk_penalty = risk_aversion * cp.quad_form(weights, cov_matrix)
    obj = profit_sum - trade_penalty - risk_penalty

    # 定义问题
    problem = cp.Problem(cp.Maximize(obj), constraints)
    # 求解
    try:
        problem.solve(solver='ECOS', qcp=True, max_iters=2000)
    except:
        problem.solve()

    # 获取解的值并返回
    # print(trade_penalty.value, risk_penalty.value)
    weights_value = weights.value
    weights_value = pd.concat([stock_today[['security_code']].reset_index(drop=True),
                               pd.DataFrame(weights_value[:, 0], columns=['weight']).reset_index(drop=True)],
                              axis=1)
    weights_value.loc[weights_value['weight'] < 0.001, 'weight'] = 0
    weights_value = weights_value[weights_value['weight'] != 0]
    weights_value['weight'] = weights_value['weight'] / weights_value['weight'].sum()
    weights_value = weights_value.set_index('security_code')['weight'].to_dict()
    # print(trade_cost_sum.value)
    return weights_value, problem.status


def trade_cal(past_weight, holding_df, trade_target, today_stock_return, drop_list=[]):

    weight_diff = pd.merge(left=past_weight, right=trade_target, on='security_code', how='outer')
    weight_diff.fillna(0, inplace=True)
    weight_diff = weight_diff[(weight_diff['past_weight'] != 0) | (weight_diff['trade_weight'] != 0)]  # 有交易的
    weight_diff = pd.merge(weight_diff, today_stock_return, on=['security_code'], how='left')
    weight_diff['diff'] = weight_diff['trade_weight'] - weight_diff['past_weight']
    weight_diff.loc[abs(weight_diff['diff']) < 0.001, 'diff'] = 0  # 差距过小的不进行交易
    weight_diff.loc[weight_diff['trade_weight'] == 0, 'diff'] = \
        -weight_diff.loc[weight_diff['trade_weight'] == 0, 'past_weight']
    # 部分个股长时间停牌没有价格数据，因此填充成交额为0，价格为1，由于都是百分比计算，因此只要让open和close均为1，即可保持数值不变
    weight_diff['next_tvol'].fillna(0, inplace=True)
    weight_diff.fillna(1, inplace=True)
    weight_diff.reset_index(inplace=True)
    del weight_diff['index']
    sell_list = list(weight_diff.loc[np.where((weight_diff['diff'] < 0)
                                              & (weight_diff['next_tvol'] != 0)), 'security_code'])
    buy_list = list(weight_diff.loc[np.where((weight_diff['diff'] > 0)
                                             & (weight_diff['next_tvol'] != 0)), 'security_code'])
    stable_list = [stock_ for stock_ in holding_df.index if stock_ not in sell_list + buy_list]
    stable_list.remove('cash')
    for code_ in drop_list:
        drop_list.remove(code_)
    past_value = holding_df['past_value'].sum()  # 前一天持有的所有资产，因为要提前做交易单，所以按前一天净值算
    # 未停牌的能卖出的按开盘价结算卖出
    total_sell = 0
    for stock_ in sell_list:
        stock_info = weight_diff[weight_diff['security_code'] == stock_]
        next_holding = holding_df.loc[stock_, 'past_value'] / stock_info['s_dq_adjclose'].iloc[0] \
                       * stock_info['next_open'].iloc[0]  # 转日开盘后个股价值
        holding_df.loc['cash', 'past_value'] += -next_holding * stock_info['diff'].iloc[0] \
                                                / stock_info['past_weight'].iloc[0] \
                                                * (1 - mc.trade_cost - mc.sell_fee)  # 卖出部分变现
        holding_df.loc[stock_, 'past_value'] = next_holding * stock_info['trade_weight'].iloc[0] \
                                                / stock_info['past_weight'].iloc[0]  # 持仓改变
        total_sell += -next_holding * stock_info['diff'].iloc[0] \
                                                / stock_info['past_weight'].iloc[0] * (1 - mc.trade_cost - mc.sell_fee)
    for stock_ in drop_list:  # 直接被收购退市的，按价值卖出
        holding_df.loc['cash', 'past_value'] += holding_df.loc[stock_, 'past_value'] * (1 - mc.trade_cost - mc.sell_fee)
        holding_df.loc[stock_, 'past_value'] = 0
        total_sell += holding_df.loc[stock_, 'past_value'] * (1 - mc.trade_cost - mc.sell_fee)
    # 剩余的钱且未停牌的按weight重置进行买入
    buy_df = weight_diff[weight_diff['security_code'].isin(buy_list)]['diff'].sum()
    buy_cash = holding_df.loc['cash'].iloc[0]  # 可以买的钱

    total_buy = 0
    for stock_ in buy_list:
        stock_info = weight_diff[weight_diff['security_code'] == stock_]
        buy_asset_total = past_value * stock_info['diff'].iloc[0]  # 按总资产计算的购买数量
        buy_asset_cash = buy_cash * stock_info['diff'].iloc[0] / buy_df  # 按现有资金计算的购买量
        buy_asset = min(buy_asset_cash, buy_asset_total)
        holding_df.loc['cash', 'past_value'] -= buy_asset
        if stock_info['past_weight'].iloc[0] == 0:  # 昨日没持仓
            holding_df.loc[stock_, 'past_value'] = buy_asset * (1 - mc.trade_cost - mc.buy_fee)
        else:  # 昨天有持仓
            holding_df.loc[stock_, 'past_value'] = holding_df.loc[stock_, 'past_value'] \
                                                   / stock_info['s_dq_adjclose'].iloc[0] \
                                                   * stock_info['next_open'].iloc[0] \
                                                   + buy_asset * (1 - mc.trade_cost - mc.buy_fee)
        total_buy += buy_asset

    # 无操作个股按收盘、开盘结算
    for stock_ in stable_list:
        stock_info = weight_diff[weight_diff['security_code'] == stock_]
        holding_df.loc[stock_, 'past_value'] = holding_df.loc[stock_, 'past_value'] \
                                               / stock_info['s_dq_adjclose'].iloc[0] \
                                               * stock_info['next_open'].iloc[0]  # 转日开盘后个股价值

    # 更新持仓，去除0的
    holding_df = holding_df[holding_df['past_value'] != 0]
    if 'cash' not in holding_df.index:
        holding_df.loc['cash'] = 0
    # holding_df.dropna(inplace=True)

    # 尾盘结算价格
    holding_df.reset_index(inplace=True)
    holding_df = pd.merge(left=holding_df, right=weight_diff[['security_code', 'next_open', 'next_close']],
                          on=['security_code'], how='left')
    holding_df.loc[holding_df['security_code'] == 'cash', ['next_open', 'next_close']] = 1  # Cash项价格不变
    holding_df['past_value'] = holding_df['past_value'] * holding_df['next_close'] / holding_df['next_open']
    del holding_df['next_close'], holding_df['next_open']
    holding_df.set_index('security_code', inplace=True)
    turnover_rate = (total_buy + total_sell) / past_value / 2

    print(today_stock_return['trade_date'].iloc[0], turnover_rate)

    return holding_df, turnover_rate


# stock_today = score_df[score_df['trade_date'] == date_]
# stock_today[stock_today['security_code'] == '600650']

def backtest_single(score_df, turnover_punish, file_name):

    # 个股数据整理
    # 生成个股和指数涨跌幅
    stock_return = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_return['open'] = stock_return['s_dq_open'] * stock_return['s_dq_adjclose'] / stock_return['s_dq_close']  # 后复权后开盘价
    stock_return['next_open'] = stock_return.groupby('security_code')['open'].shift(-1)
    stock_return['next_close'] = stock_return.groupby('security_code')['s_dq_adjclose'].shift(-1)
    stock_return['next_tvol'] = stock_return.groupby('security_code')['tvol'].shift(-1)
    chosen_index = jc.benchmark
    index_return = pd.read_pickle(os.path.join(jc.database_path, jc.index_price))
    index_return = index_return[index_return.security_code == chosen_index]

    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_return[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    # barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    barra = pd.read_pickle(os.path.join(jc.database_path, 'barra_factor.pkl'))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    # 逐级拆分预期收益率
    init_date = score_df['trade_date'].min()
    # end_date = score_df['trade_date'].max()
    end_date = jc.end_date
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.fillna(-999, inplace=True)  #  不在选股池内，填负值
    score_df = score_df[score_df['sw_industry'] != -999]
    score_df = score_df[(score_df['trade_date'] >= init_date) & (score_df['trade_date'] <= end_date)]
    # 指数数据整理
    # 生成指数成分股的barra和行业
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 去权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)
    for date_, df_ in index_weight.groupby('trade_date'):
        df_ = df_[df_['weight'] != 0]
        df_[barra_list] = df_[barra_list].T.mul(list(df_['weight'])).T * len(df_)
        df_index.loc[date_, barra_list] = df_[barra_list].mean()  # 指数barra
        df_index.loc[date_, barra_std_list] = list(df_[barra_list].std())  # 指数barra标准差
        df_index.loc[date_, industry_list] = df_['sw_industry'].value_counts() / len(df_)
    df_index.fillna(0, inplace=True)

    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 逐日回测（包含组合优化）
    date_list = score_df['trade_date'].drop_duplicates().sort_values().to_list()
    opt_weight = {}  # 字典形式储存结果
    initial_asset = 1000
    asset_record = pd.DataFrame(columns=['asset'])  # 纪录每日净值
    holding_record = {}  # 持仓权重，尾盘更新
    holding_record[date_list[0]] = pd.DataFrame({'security_code':['cash'],
                                                 'past_value':[initial_asset]}).set_index('security_code')
    turnover_record = pd.DataFrame(columns=['turnover_rate'])  # 纪录换手率
    holding_value_record = pd.DataFrame(columns=['num', 'value'])  # 记录持仓数量和平均市值
    # 记录Barra持仓
    barra_beta = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_momentum = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_size = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_earnyild = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_resvol = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_growth = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_btop = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_leverage = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_liquidty = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_sizenl = pd.DataFrame(columns=['index', 'diff', 'barra'])
    # 读取市值
    stock_value = pd.read_pickle(os.path.join(jc.database_path, jc.mkt_capt_data))
    excel_df = []
    holding_index_weight = pd.DataFrame(columns=['300', '500', '1000', 'else'])
    stock_index_weight = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    index_stock_300 = stock_index_weight[stock_index_weight['index_code'] == '000300']
    index_stock_500 = stock_index_weight[stock_index_weight['index_code'] == '000905']
    index_stock_1000 = stock_index_weight[stock_index_weight['index_code'] == '000852']
    stock_index_date_300 = list(index_stock_300['trade_date'].unique())
    stock_index_date_500 = list(index_stock_500['trade_date'].unique())
    stock_index_date_1000 = list(index_stock_1000['trade_date'].unique())
    stock_index_date_300.sort()
    stock_index_date_500.sort()
    stock_index_date_1000.sort()
    for date_num in range(0, len(date_list) - 1):
    # for date_num in range(0, 598):
        date_ = date_list[date_num]  # 1656崩了, date_num=598
        next_date = date_list[date_num + 1]
        stock_today = score_df[score_df['trade_date'] == date_]
        stock_today.drop_duplicates(inplace=True)
        index_today = df_index.loc[date_]
        holding_df = holding_record[date_]
        past_weight = holding_df.reset_index()  # 根据净值计算权重
        past_weight['past_value'] = past_weight['past_value'] / past_weight['past_value'].sum()
        past_weight.columns = ['security_code', 'past_weight']
        past_weight = past_weight[past_weight['security_code'] != 'cash']
        past_weight = pd.merge(left=stock_today[['security_code']], right=past_weight,
                               on='security_code', how='outer')
        past_weight.fillna(0, inplace=True)
        stock_today = pd.merge(left=stock_today, right=past_weight[['security_code']],
                               on='security_code', how='outer')
        stock_today['trade_date'] = date_

        # 可能有突然被收购股
        nan_df = stock_today[pd.isna(stock_today['sw_industry'])]
        if len(nan_df) > 0:
            drop_list = list(nan_df['security_code'])
            stock_today_drop = stock_today[~stock_today['security_code'].isin(drop_list)]
            past_weight_drop = past_weight[~past_weight['security_code'].isin(drop_list)]
            # 计算当日opt_weight, 无解情况下则放宽限制
            barra_std_num = mc.barra_std_num
            try:
                opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight_drop,
                                                     stock_today_drop, index_today, turnover_punish, turnover=0.15)
            except:
                try:
                    opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight_drop,
                                                         stock_today_drop, index_today, turnover_punish, turnover=0.2)
                except:
                    barra_std_num += mc.barra_std_num / 2
                    for barra_ in barra_list:
                        df_index.loc[date_, barra_ + '_lower'] = \
                            (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
                        df_index.loc[date_, barra_ + '_upper'] = \
                            (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
                    index_today = df_index.loc[date_]
                    opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight_drop,
                                                         stock_today_drop, index_today, turnover_punish, turnover=0.2)
            turnover = 0.2
            while status not in ['optimal', 'optimal_inaccurate']:
                turnover += 0.05
                opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight_drop,
                                                     stock_today_drop, index_today, turnover_punish, turnover=turnover)
            # 进行交易
            trade_target = pd.DataFrame(opt_weight[date_].values(), index=opt_weight[date_].keys()).reset_index()
            trade_target.columns = ['security_code', 'trade_weight']
            nan_df = pd.DataFrame({'security_code':drop_list})
            nan_df['trade_weight'] = 0
            trade_target = pd.concat([trade_target, nan_df], axis=0)
            today_stock_return = stock_return[stock_return['trade_date'] == date_]
            holding_df, turnover_rate = trade_cal(past_weight, holding_df, trade_target, today_stock_return, drop_list)
        else:
            # 计算当日opt_weight, 无解情况下则放宽限制
            barra_std_num = mc.barra_std_num
            try:
                opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                     stock_today, index_today, turnover_punish, turnover=0.15)
            except:
                try:
                    opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                         stock_today, index_today, turnover_punish, turnover=0.2)
                except:
                    barra_std_num += mc.barra_std_num / 2
                    for barra_ in barra_list:
                        df_index.loc[date_, barra_ + '_lower'] = \
                            (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
                        df_index.loc[date_, barra_ + '_upper'] = \
                            (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
                    index_today = df_index.loc[date_]
                    opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                         stock_today, index_today, turnover_punish, turnover=0.2)
            turnover = 0.2
            while status not in ['optimal', 'optimal_inaccurate']:
                turnover += 0.05
                opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                     stock_today, index_today, turnover_punish, turnover=turnover)
            # 进行交易
            trade_target = pd.DataFrame(opt_weight[date_].values(), index=opt_weight[date_].keys()).reset_index()
            trade_target.columns = ['security_code', 'trade_weight']
            today_stock_return = stock_return[stock_return['trade_date'] == date_]
            holding_df, turnover_rate = trade_cal(past_weight, holding_df, trade_target, today_stock_return)
        # 计算成分股比例
        holding_index = holding_df.copy()
        holding_index['weight'] = holding_index['past_value'] / holding_index['past_value'].sum()
        weight_300 = 0
        weight_500 = 0
        weight_1000 = 0
        weight_else = 0
        last_300_date = [date_ for date_ in stock_index_date_300 if date_ <= next_date][-1]
        last_500_date = [date_ for date_ in stock_index_date_500 if date_ <= next_date][-1]
        last_1000_date = [date_ for date_ in stock_index_date_1000 if date_ <= next_date][-1]
        index_stock_300_today = index_stock_300[index_stock_300['trade_date'] == last_300_date]
        index_stock_500_today = index_stock_500[index_stock_500['trade_date'] == last_500_date]
        index_stock_1000_today = index_stock_1000[index_stock_1000['trade_date'] == last_1000_date]
        for stock_ in holding_index.index:
            if stock_ in list(index_stock_300_today['security_code']):
                weight_300 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_500_today['security_code']):
                weight_500 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_1000_today['security_code']):
                weight_1000 += holding_index.loc[stock_, 'weight']
            else:
                weight_else += holding_index.loc[stock_, 'weight']
        holding_index_weight.loc[next_date] = [weight_300, weight_500, weight_1000, weight_else]
        # 净值和持仓记录
        holding_record[next_date] = holding_df
        asset_record.loc[next_date] = holding_df['past_value'].sum() / initial_asset
        turnover_record.loc[next_date] = turnover_rate
        today_mkt = pd.DataFrame(holding_df).reset_index()
        today_mkt['trade_date'] = next_date
        today_stock_capt = stock_value[stock_value['trade_date'] == next_date][['security_code',
                                                                                'trade_date', 'total_capt']]
        today_mkt = pd.merge(left=today_mkt, right=today_stock_capt, on=['security_code', 'trade_date'], how='inner')
        # mkt_value = (today_mkt['free_capt'] * today_mkt['past_value'] / today_mkt['past_value'].sum()).sum() / 10000
        mkt_value = today_mkt['total_capt'].median() / 10000
        holding_value_record.loc[next_date] = [len(today_mkt), mkt_value]
        excel_df.append(today_mkt)
        # 计算barra暴露
        today_barra = pd.DataFrame(holding_df).reset_index()
        today_barra['trade_date'] = next_date
        today_barra_all = barra[barra['trade_date'] == next_date]
        today_barra = pd.merge(left=today_barra, right=today_barra_all, on=['security_code', 'trade_date'], how='inner')
        today_barra[barra_list] = today_barra[barra_list] * today_barra['past_value'].values.reshape(-1, 1) \
                                  / today_mkt['past_value'].sum()
        index_next = df_index.loc[next_date]
        diff = (today_barra['barra_beta'].sum() - index_next['barra_beta']) / (
                    (index_next['barra_beta_upper'] - index_next['barra_beta']) / mc.barra_std_num)
        barra_beta.loc[next_date] = [index_next['barra_beta'], diff, today_barra['barra_beta'].sum()]
        diff = (today_barra['barra_momentum'].sum() - index_next['barra_momentum']) / (index_next['barra_momentum_std'])
        barra_momentum.loc[next_date] = [index_next['barra_momentum'], diff, today_barra['barra_momentum'].sum()]
        diff = (today_barra['barra_size'].sum() - index_next['barra_size']) / (index_next['barra_size_std'])
        barra_size.loc[next_date] = [index_next['barra_size'], diff, today_barra['barra_size'].sum()]
        diff = (today_barra['barra_earnyild'].sum() - index_next['barra_earnyild']) / (
                (index_next['barra_earnyild_upper'] - index_next['barra_earnyild']) / mc.barra_std_num)
        barra_earnyild.loc[next_date] = [index_next['barra_earnyild'], diff, today_barra['barra_earnyild'].sum()]
        diff = (today_barra['barra_resvol'].sum() - index_next['barra_resvol']) / (
                (index_next['barra_resvol_upper'] - index_next['barra_resvol']) / mc.barra_std_num)
        barra_resvol.loc[next_date] = [index_next['barra_resvol'], diff, today_barra['barra_resvol'].sum()]
        diff = (today_barra['barra_growth'].sum() - index_next['barra_growth']) / (
                (index_next['barra_growth_upper'] - index_next['barra_growth']) / mc.barra_std_num)
        barra_growth.loc[next_date] = [index_next['barra_growth'], diff, today_barra['barra_growth'].sum()]
        diff = (today_barra['barra_btop'].sum() - index_next['barra_btop']) / (
                (index_next['barra_btop_upper'] - index_next['barra_btop']) / mc.barra_std_num)
        barra_btop.loc[next_date] = [index_next['barra_btop'], diff, today_barra['barra_btop'].sum()]
        diff = (today_barra['barra_leverage'].sum() - index_next['barra_leverage']) / (
                (index_next['barra_leverage_upper'] - index_next['barra_leverage']) / mc.barra_std_num)
        barra_leverage.loc[next_date] = [index_next['barra_leverage'], diff, today_barra['barra_leverage'].sum()]
        diff = (today_barra['barra_liquidty'].sum() - index_next['barra_liquidty']) / (
                (index_next['barra_liquidty_upper'] - index_next['barra_liquidty']) / mc.barra_std_num)
        barra_liquidty.loc[next_date] = [index_next['barra_liquidty'], diff, today_barra['barra_liquidty'].sum()]
        try:
            diff = (today_barra['barra_sizenl'].sum() - index_next['barra_sizenl']) / (index_next['barra_sizenl_std'])
            barra_sizenl.loc[next_date] = [index_next['barra_sizenl'], diff, today_barra['barra_sizenl'].sum()]
        except:
            diff = (today_barra['barra_midcap'].sum() - index_next['barra_midcap']) / (index_next['barra_midcap'])
            barra_sizenl.loc[next_date] = [index_next['barra_midcap'], diff, today_barra['barra_midcap'].sum()]

    # excel_df = pd.concat(excel_df, axis=0)
    # excel_df.to_excel('回测监控.xlsx')

    holding_result = []
    for date_ in holding_record.keys():
        holding_ = holding_record[date_].reset_index()
        holding_['trade_date'] = date_
        holding_result.append(holding_)
    holding_result = pd.concat(holding_result, axis=0)
    holding_result.to_excel('holding_result.xlsx')
    # 生成回测文档
    asset_record.loc[asset_record.index[0] - timedelta(days=1), 'asset'] = 1  # 加入初始日期
    asset_record.sort_index(inplace=True)
    asset_record.reset_index(inplace=True)
    asset_record.columns = ['trade_date', 'origin']
    index_init = index_return[index_return['trade_date'] <= asset_record['trade_date'].iloc[0]]['close'].iloc[-1]
    asset_record = pd.merge(left=asset_record, right=index_return[['trade_date', 'close']], on=['trade_date'], how='left')
    asset_record.fillna(index_init, inplace=True)  # 有可能指数在首日没有数值
    asset_record['close'] = asset_record['close'] / index_init
    asset_record['asset'] = asset_record['origin'] / asset_record['close']

    asset_record['return_rate'] = asset_record['asset'].pct_change()
    asset_record['max_draw_down'] = asset_record['asset'].expanding().max()  # 累计收益率的历史最大值
    asset_record['max_draw_down'] = asset_record['asset'] / asset_record['max_draw_down'] - 1  # 回撤
    asset_record['trade_year'] = asset_record['trade_date'].dt.year
    asset_record.to_excel('asset_record.xlsx')
    # 图1：绝对收益、指数收益、超额收益
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为1天
    ax2.bar(list(asset_record['trade_date']), list(asset_record['asset'] - 1), width=bar_width,
            color='skyblue', label='累计超额收益', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['origin']), color='red', label='策略绝对净值')
    ax1.plot(list(asset_record['trade_date']), list(asset_record['close']), color='black', label='指数净值')
    plt.title('绝对收益')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r".//abs_asset.jpg", dpi=1000)

    # 图2： 超额收益、超额最大回撤
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为xx天 len(date_list)/50+1
    ax2.bar(asset_record['trade_date'], asset_record['max_draw_down'], width=bar_width,
            color='skyblue', label='最大回撤', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['asset']), color='red', label='净值走势')
    plt.title('超额收益及超额最大回撤')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r"./figure_save/increase_asset.jpg", dpi=1000)
    # 图3：换手率走势
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    turnover_figure = turnover_record.iloc[1:]  # 第一天建仓，数据没意义
    ax1.plot(list(turnover_figure.index), list(turnover_figure['turnover_rate']), color='red', label='换手率走势')
    plt.title('换手率')
    lines, labels = ax1.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/turnover_rate.jpg", dpi=1000)
    # 图4：平均市值变动
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(list(holding_value_record.index), list(holding_value_record['num']), color='red', label='持仓个股数目')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('持仓个股数目', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(list(holding_value_record.index), list(holding_value_record['value']), color='blue', label='总市值中位数')
    ax2.set_ylabel('总市值中位数', color='blue')
    ax2.tick_params('y', colors='blue')
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines])
    plt.title('持仓数量和市值')
    plt.savefig(r"./figure_save/holding_value.jpg", dpi=1000)
    # plt.savefig(os.path.join(file_name, 'holding_value.jpg'), dpi=1000)
    # 图5：持仓股比例
    plt.figure(figsize=(10, 6))
    ax = holding_index_weight.plot(kind='bar', stacked=True)
    holding_index_num = holding_index_weight.copy()
    holding_index_num['year'] = holding_index_weight.index.year
    holding_index_num.reset_index(inplace=True)
    xticks = list(holding_index_num.drop_duplicates(subset='year').index)
    xlabels = [holding_index_weight.index[x].strftime('%Y') for x in xticks]
    plt.xticks(rotation=0)
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('指数成分股占比')
    plt.show()
    plt.savefig(r"./figure_save/holding_index_weight.jpg", dpi=1000)
    # 图5-15：barra图
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_beta
    name = 'beta'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势'%name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限'%name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限'%name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离'%name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势'%name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg"%name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_momentum
    name = 'momentum'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_size
    name = 'size'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_earnyild
    name = 'earnyild'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_resvol
    name = 'resvol'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_growth
    name = 'growth'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_btop
    name = 'btop'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_leverage
    name = 'leverage'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_liquidty
    name = 'liquidty'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_sizenl
    name = 'sizenl'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    # 逐年收益、夏普、回撤
    return_yr = asset_record.groupby(['trade_year']).last().reset_index()
    yr_days = asset_record.groupby(['trade_year']).size().reset_index()
    yr_days.columns = ['trade_year', 'days']
    return_yr = pd.merge(return_yr, yr_days, on=['trade_year'], how='left')
    return_yr['asset_yr'] = return_yr['asset'] / return_yr['asset'].shift(1).fillna(1)
    return_yr['return_yr'] = round(return_yr['asset_yr'] ** (252 / return_yr['days']) - 1, 5)
    sharpe_yr = ((asset_record.groupby(['trade_year'])['return_rate'].mean()
                  / asset_record.groupby(['trade_year'])['return_rate'].std()) * np.sqrt(252)).reset_index()
    sharpe_yr.columns = ['trade_year', 'sharpe_yr']
    max_drawdown_yr = (asset_record.groupby(['trade_year'])['max_draw_down'].min()).reset_index()
    yr = pd.merge(return_yr[['trade_year', 'return_yr']], sharpe_yr, on=['trade_year'], how='left')
    yr = pd.merge(yr, max_drawdown_yr, on=['trade_year'], how='left')

    # 总体收益、夏普、回撤
    return_all = asset_record.tail(1)
    all_days = asset_record.shape[0]
    return_all['return_all'] = round(return_all['asset'] ** (252 / all_days) - 1, 5)
    sharpe_all = ((asset_record['return_rate'].mean()
                   / asset_record['return_rate'].std()) * np.sqrt(252))
    max_drawdown_all = (asset_record['max_draw_down'].min())
    all_record = pd.DataFrame({'return_all': [return_all['return_all'].iloc[0]], 'sharpe_all': sharpe_all,
                         'max_drawdown_all': max_drawdown_all})

    # # 输出文档
    # test_doc = Document()
    # style = 'Light List Accent 2'
    # default_section = test_doc.sections[0]
    # default_section.page_width = Cm(30)  # 纸张大小改为自定义，方便放下大表和大图
    # test_doc.styles['Normal'].font.name = 'Times New Roman'
    # test_doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
    # title = date_list[0].strftime('%Y-%m-%d') + '至' + date_list[-1].strftime('%Y-%m-%d') + '回测'
    # add_heading(title, test_doc, level=0, seq=1)
    # add_heading('整体业绩表现', test_doc, level=1, seq=1)
    # all_record.columns = ['超额年化收益率', '超额夏普比率', '超额最大回撤']
    # all_record['超额年化收益率'] = all_record.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    # all_record['超额夏普比率'] = all_record.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    # all_record['超额最大回撤'] = all_record.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)
    # add_table(all_record.T.reset_index().T, test_doc, size=7, style=style, width=8)

    # images = os.path.join(r'./figure_save/abs_asset.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/increase_asset.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/turnover_rate.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/holding_value.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/holding_index_weight.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/beta.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/momentum.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/size.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/earnyild.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/resvol.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/growth.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/btop.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/leverage.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/liquidty.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # images = os.path.join(r'./figure_save/sizenl.jpg')
    # doc_add_images(test_doc, images, inch=6)
    # add_heading('逐年业绩表现', test_doc, level=1, seq=2)
    # num = 0  # 初始计数
    # yr.columns = ['年份', '超额年化收益率', '超额夏普比率', '超额最大回撤']
    # yr['年份'] = yr['年份'].astype('str')
    # yr['超额年化收益率'] = yr.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    # yr['超额夏普比率'] = yr.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    # yr['超额最大回撤'] = yr.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)

    # for year_ in asset_record['trade_year'].unique():
    #     num += 1
    #     add_heading('%s业绩表现'%year_, test_doc, level=2, seq=num)
    #     table_ = yr[yr['年份'] == str(year_)]
    #     add_table(table_.T.reset_index().T, test_doc, size=7, style=style, width=8)
    # test_doc.save(r"./%s.docx"%file_name)


def backtest_single_cov(score_df, turnover_punish, file_name):

    # 个股数据整理
    # 生成个股和指数涨跌幅
    stock_return = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_return['open'] = stock_return['s_dq_open'] * stock_return['s_dq_adjclose'] / stock_return['s_dq_close']  # 后复权后开盘价
    stock_return['next_open'] = stock_return.groupby('security_code')['open'].shift(-1)
    stock_return['next_close'] = stock_return.groupby('security_code')['s_dq_adjclose'].shift(-1)
    stock_return['next_tvol'] = stock_return.groupby('security_code')['tvol'].shift(-1)
    stock_return['last_close'] = stock_return.groupby('security_code')['s_dq_adjclose'].shift(1)
    stock_return['today_chg'] = stock_return['s_dq_adjclose'] / stock_return['last_close'] - 1
    chosen_index = jc.benchmark
    index_return = pd.read_pickle(os.path.join(jc.database_path, jc.index_price))
    index_return = index_return[index_return.security_code == chosen_index]

    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_return[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    # 逐级拆分预期收益率
    init_date = score_df['trade_date'].min()
    end_date = score_df['trade_date'].max()
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.fillna(-999, inplace=True)  #  不在选股池内，填负值
    score_df = score_df[(score_df['trade_date'] >= init_date) & (score_df['trade_date'] <= end_date)]
    # 指数数据整理
    # 生成指数成分股的barra和行业
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 去权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)
    for date_, df_ in index_weight.groupby('trade_date'):
        df_ = df_[df_['weight'] != 0]
        df_[barra_list] = df_[barra_list].T.mul(list(df_['weight'])).T * len(df_)
        df_index.loc[date_, barra_list] = df_[barra_list].mean()  # 指数barra
        df_index.loc[date_, barra_std_list] = list(df_[barra_list].std())  # 指数barra标准差
        df_index.loc[date_, industry_list] = df_['sw_industry'].value_counts() / len(df_)
    df_index.fillna(0, inplace=True)

    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 逐日回测（包含组合优化）
    date_list = score_df['trade_date'].drop_duplicates().sort_values().to_list()
    opt_weight = {}  # 字典形式储存结果
    initial_asset = 1000
    asset_record = pd.DataFrame(columns=['asset'])  # 纪录每日净值
    holding_record = {}  # 持仓权重，尾盘更新
    holding_record[date_list[0]] = pd.DataFrame({'security_code':['cash'],
                                                 'past_value':[initial_asset]}).set_index('security_code')
    turnover_record = pd.DataFrame(columns=['turnover_rate'])  # 纪录换手率
    holding_value_record = pd.DataFrame(columns=['num', 'value'])  # 记录持仓数量和平均市值
    # 记录Barra持仓
    barra_beta = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_momentum = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_size = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_earnyild = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_resvol = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_growth = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_btop = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_leverage = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_liquidty = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_sizenl = pd.DataFrame(columns=['index', 'diff', 'barra'])
    # 读取市值
    stock_value = pd.read_pickle(os.path.join(jc.database_path, jc.mkt_capt_data))
    excel_df = []
    holding_index_weight = pd.DataFrame(columns=['300', '500', '1000', 'else'])
    stock_index_weight = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    index_stock_300 = stock_index_weight[stock_index_weight['index_code'] == '000300']
    index_stock_500 = stock_index_weight[stock_index_weight['index_code'] == '000905']
    index_stock_1000 = stock_index_weight[stock_index_weight['index_code'] == '000852']
    stock_index_date_300 = list(index_stock_300['trade_date'].unique())
    stock_index_date_500 = list(index_stock_500['trade_date'].unique())
    stock_index_date_1000 = list(index_stock_1000['trade_date'].unique())
    stock_index_date_300.sort()
    stock_index_date_500.sort()
    stock_index_date_1000.sort()
    cov_date_list = stock_return['trade_date'].drop_duplicates().sort_values().to_list()
    for date_num in range(0, len(date_list) - 1):
    # for date_num in range(0, 40):
        # date_ = pd.to_datetime('2022-12-30')
        date_ = date_list[date_num]
        next_date = date_list[date_num + 1]
        stock_today = score_df[score_df['trade_date'] == date_]
        stock_today.drop_duplicates(inplace=True)
        index_today = df_index.loc[date_]
        holding_df = holding_record[date_]
        past_weight = holding_df.reset_index()  # 根据净值计算权重
        past_weight['past_value'] = past_weight['past_value'] / past_weight['past_value'].sum()
        past_weight.columns = ['security_code', 'past_weight']
        past_weight = past_weight[past_weight['security_code'] != 'cash']
        past_weight = pd.merge(left=stock_today[['security_code']], right=past_weight,
                               on='security_code', how='outer')
        past_weight.fillna(0, inplace=True)
        stock_today = pd.merge(left=stock_today, right=past_weight[['security_code']],
                               on='security_code', how='outer')
        stock_today['trade_date'] = date_

        # 计算当日opt_weight, 无解情况下则放宽限制
        barra_std_num = mc.barra_std_num
        last_cov_date = [trade_date_ for trade_date_ in cov_date_list if trade_date_ <= date_][-20]
        cov_df = stock_return[(stock_return['trade_date'] >= last_cov_date) & (stock_return['trade_date'] <= date_)]
        cov_df = cov_df[cov_df['security_code'].isin(past_weight['security_code'])]
        cov_df = cov_df.pivot(index='trade_date', columns='security_code', values='today_chg')
        cov_matrix = np.cov(cov_df, rowvar=False)
        cov_matrix = np.nan_to_num(cov_matrix)

        try:
            opt_weight[date_], status = port_opt_cov(['chg'], industry_list, barra_list, past_weight,
                                                     stock_today, index_today, turnover_punish, cov_matrix, turnover=0.15)
        except:
            try:
                opt_weight[date_], status = port_opt_cov(['chg'], industry_list, barra_list, past_weight,
                                                         stock_today, index_today, turnover_punish, cov_matrix, turnover=0.2)
            except:
                barra_std_num += mc.barra_std_num / 2
                for barra_ in barra_list:
                    df_index.loc[date_, barra_ + '_lower'] = \
                        (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
                    df_index.loc[date_, barra_ + '_upper'] = \
                        (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
                index_today = df_index.loc[date_]
                opt_weight[date_], status = port_opt_cov(['chg'], industry_list, barra_list, past_weight,
                                                         stock_today, index_today, turnover_punish, cov_matrix, turnover=0.2)
        turnover = 0.2
        while status not in ['optimal', 'optimal_inaccurate']:
            try:
                turnover += 0.05
                opt_weight[date_], status = port_opt_cov(['chg'], industry_list, barra_list, past_weight,
                                                         stock_today, index_today, turnover_punish, cov_matrix, turnover=turnover)
            except:
                opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                     stock_today, index_today, turnover_punish, turnover=0.15)

        # while status not in ['optimal', 'optimal_inaccurate']:
        #     try:
        #         barra_std_num += mc.barra_std_num / 2
        #         for barra_ in barra_list:
        #             df_index.loc[date_, barra_ + '_lower'] = (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
        #             df_index.loc[date_, barra_ + '_upper'] = (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
        #         index_today = df_index.loc[date_]
        #         opt_weight[date_], status = port_opt_cov(['chg'], industry_list, barra_list, past_weight,
        #                                              stock_today, index_today, turnover_punish)
        #     except:
        #         opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
        #                                              stock_today, index_today, turnover_punish)

        # 进行交易
        trade_target = pd.DataFrame(opt_weight[date_].values(), index=opt_weight[date_].keys()).reset_index()
        trade_target.columns = ['security_code', 'trade_weight']
        today_stock_return = stock_return[stock_return['trade_date'] == date_]
        holding_df, turnover_rate = trade_cal(past_weight, holding_df, trade_target, today_stock_return)
        # 计算成分股比例
        holding_index = holding_df.copy()
        holding_index['weight'] = holding_index['past_value'] / holding_index['past_value'].sum()
        weight_300 = 0
        weight_500 = 0
        weight_1000 = 0
        weight_else = 0
        last_300_date = [date_ for date_ in stock_index_date_300 if date_ <= next_date][-1]
        last_500_date = [date_ for date_ in stock_index_date_500 if date_ <= next_date][-1]
        last_1000_date = [date_ for date_ in stock_index_date_1000 if date_ <= next_date][-1]
        index_stock_300_today = index_stock_300[index_stock_300['trade_date'] == last_300_date]
        index_stock_500_today = index_stock_500[index_stock_500['trade_date'] == last_500_date]
        index_stock_1000_today = index_stock_1000[index_stock_1000['trade_date'] == last_1000_date]
        for stock_ in holding_index.index:
            if stock_ in list(index_stock_300_today['security_code']):
                weight_300 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_500_today['security_code']):
                weight_500 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_1000_today['security_code']):
                weight_1000 += holding_index.loc[stock_, 'weight']
            else:
                weight_else += holding_index.loc[stock_, 'weight']
        holding_index_weight.loc[next_date] = [weight_300, weight_500, weight_1000, weight_else]
        # 净值和持仓记录
        holding_record[next_date] = holding_df
        asset_record.loc[next_date] = holding_df['past_value'].sum() / initial_asset
        turnover_record.loc[next_date] = turnover_rate
        today_mkt = pd.DataFrame(holding_df).reset_index()
        today_mkt['trade_date'] = next_date
        today_stock_capt = stock_value[stock_value['trade_date'] == next_date][['security_code',
                                                                                'trade_date', 'total_capt']]
        today_mkt = pd.merge(left=today_mkt, right=today_stock_capt, on=['security_code', 'trade_date'], how='inner')
        # mkt_value = (today_mkt['free_capt'] * today_mkt['past_value'] / today_mkt['past_value'].sum()).sum() / 10000
        mkt_value = today_mkt['total_capt'].median() / 10000
        holding_value_record.loc[next_date] = [len(today_mkt), mkt_value]
        excel_df.append(today_mkt)
        # 计算barra暴露
        today_barra = pd.DataFrame(holding_df).reset_index()
        today_barra['trade_date'] = next_date
        today_barra_all = barra[barra['trade_date'] == next_date]
        today_barra = pd.merge(left=today_barra, right=today_barra_all, on=['security_code', 'trade_date'], how='inner')
        today_barra[barra_list] = today_barra[barra_list] * today_barra['past_value'].values.reshape(-1, 1) \
                                  / today_mkt['past_value'].sum()
        index_next = df_index.loc[next_date]
        diff = (today_barra['barra_beta'].sum() - index_next['barra_beta']) / (
                    (index_next['barra_beta_upper'] - index_next['barra_beta']) / mc.barra_std_num)
        barra_beta.loc[next_date] = [index_next['barra_beta'], diff, today_barra['barra_beta'].sum()]
        diff = (today_barra['barra_momentum'].sum() - index_next['barra_momentum']) / (
                (index_next['barra_momentum_upper'] - index_next['barra_momentum']) / mc.barra_std_num)
        barra_momentum.loc[next_date] = [index_next['barra_momentum'], diff, today_barra['barra_momentum'].sum()]
        diff = (today_barra['barra_size'].sum() - index_next['barra_size']) / (
                (index_next['barra_size_upper'] - index_next['barra_size']) / mc.barra_std_num)
        barra_size.loc[next_date] = [index_next['barra_size'], diff, today_barra['barra_size'].sum()]
        diff = (today_barra['barra_earnyild'].sum() - index_next['barra_earnyild']) / (
                (index_next['barra_earnyild_upper'] - index_next['barra_earnyild']) / mc.barra_std_num)
        barra_earnyild.loc[next_date] = [index_next['barra_earnyild'], diff, today_barra['barra_earnyild'].sum()]
        diff = (today_barra['barra_resvol'].sum() - index_next['barra_resvol']) / (
                (index_next['barra_resvol_upper'] - index_next['barra_resvol']) / mc.barra_std_num)
        barra_resvol.loc[next_date] = [index_next['barra_resvol'], diff, today_barra['barra_resvol'].sum()]
        diff = (today_barra['barra_growth'].sum() - index_next['barra_growth']) / (
                (index_next['barra_growth_upper'] - index_next['barra_growth']) / mc.barra_std_num)
        barra_growth.loc[next_date] = [index_next['barra_growth'], diff, today_barra['barra_growth'].sum()]
        diff = (today_barra['barra_btop'].sum() - index_next['barra_btop']) / (
                (index_next['barra_btop_upper'] - index_next['barra_btop']) / mc.barra_std_num)
        barra_btop.loc[next_date] = [index_next['barra_btop'], diff, today_barra['barra_btop'].sum()]
        diff = (today_barra['barra_leverage'].sum() - index_next['barra_leverage']) / (
                (index_next['barra_leverage_upper'] - index_next['barra_leverage']) / mc.barra_std_num)
        barra_leverage.loc[next_date] = [index_next['barra_leverage'], diff, today_barra['barra_leverage'].sum()]
        diff = (today_barra['barra_liquidty'].sum() - index_next['barra_liquidty']) / (
                (index_next['barra_liquidty_upper'] - index_next['barra_liquidty']) / mc.barra_std_num)
        barra_liquidty.loc[next_date] = [index_next['barra_liquidty'], diff, today_barra['barra_liquidty'].sum()]
        diff = (today_barra['barra_sizenl'].sum() - index_next['barra_sizenl']) / (
                (index_next['barra_sizenl_upper'] - index_next['barra_sizenl']) / mc.barra_std_num)
        barra_sizenl.loc[next_date] = [index_next['barra_sizenl'], diff, today_barra['barra_sizenl'].sum()]
    # excel_df = pd.concat(excel_df, axis=0)
    # excel_df.to_excel('回测监控.xlsx')
    holding_result = []
    for date_ in holding_record.keys():
        holding_ = holding_record[date_].reset_index()
        holding_['trade_date'] = date_
        holding_result.append(holding_)
    holding_result = pd.concat(holding_result, axis=0)

    # 生成回测文档
    asset_record.loc[asset_record.index[0] - timedelta(days=1), 'asset'] = 1  # 加入初始日期
    asset_record.sort_index(inplace=True)
    asset_record.reset_index(inplace=True)
    asset_record.columns = ['trade_date', 'origin']
    index_init = index_return[index_return['trade_date'] <= asset_record['trade_date'].iloc[0]]['close'].iloc[-1]
    asset_record = pd.merge(left=asset_record, right=index_return[['trade_date', 'close']], on=['trade_date'], how='left')
    asset_record.fillna(index_init, inplace=True)  # 有可能指数在首日没有数值
    asset_record['close'] = asset_record['close'] / index_init
    asset_record['asset'] = asset_record['origin'] / asset_record['close']

    asset_record['return_rate'] = asset_record['asset'].pct_change()
    asset_record['max_draw_down'] = asset_record['asset'].expanding().max()  # 累计收益率的历史最大值
    asset_record['max_draw_down'] = asset_record['asset'] / asset_record['max_draw_down'] - 1  # 回撤
    asset_record['trade_year'] = asset_record['trade_date'].dt.year

    # 图1：绝对收益、指数收益、超额收益
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为1天
    ax2.bar(list(asset_record['trade_date']), list(asset_record['asset'] - 1), width=bar_width,
            color='skyblue', label='累计超额收益', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['origin']), color='red', label='策略绝对净值')
    ax1.plot(list(asset_record['trade_date']), list(asset_record['close']), color='black', label='指数净值')
    plt.title('绝对收益')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r"./figure_save/abs_asset.jpg", dpi=1000)

    # 图2： 超额收益、超额最大回撤
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为xx天 len(date_list)/50+1
    ax2.bar(asset_record['trade_date'], asset_record['max_draw_down'], width=bar_width,
            color='skyblue', label='最大回撤', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['asset']), color='red', label='净值走势')
    plt.title('超额收益及超额最大回撤')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r"./figure_save/increase_asset.jpg", dpi=1000)
    # 图3：换手率走势
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    turnover_figure = turnover_record.iloc[1:]  # 第一天建仓，数据没意义
    ax1.plot(list(turnover_figure.index), list(turnover_figure['turnover_rate']), color='red', label='换手率走势')
    plt.title('换手率')
    lines, labels = ax1.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/turnover_rate.jpg", dpi=1000)
    # 图4：平均市值变动
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(list(holding_value_record.index), list(holding_value_record['num']), color='red', label='持仓个股数目')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('持仓个股数目', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(list(holding_value_record.index), list(holding_value_record['value']), color='blue', label='总市值中位数')
    ax2.set_ylabel('总市值中位数', color='blue')
    ax2.tick_params('y', colors='blue')
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines])
    plt.title('持仓数量和市值')
    plt.savefig(r"./figure_save/holding_value.jpg", dpi=1000)
    # 图5：持仓股比例
    plt.figure(figsize=(10, 6))
    ax = holding_index_weight.plot(kind='bar', stacked=True)
    # plt.xticks(range(len(holding_index_weight.index)), holding_index_weight.index.strftime('%Y-%m-%d'), rotation=30)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('指数成分股占比')
    plt.show()
    plt.savefig(r"./figure_save/holding_index_weight.jpg", dpi=1000)
    # 图5-15：barra图
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_beta
    name = 'beta'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势'%name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限'%name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限'%name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离'%name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势'%name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg"%name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_momentum
    name = 'momentum'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_size
    name = 'size'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_earnyild
    name = 'earnyild'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_resvol
    name = 'resvol'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_growth
    name = 'growth'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_btop
    name = 'btop'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_leverage
    name = 'leverage'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_liquidty
    name = 'liquidty'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_sizenl
    name = 'sizenl'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    # 逐年收益、夏普、回撤
    return_yr = asset_record.groupby(['trade_year']).last().reset_index()
    yr_days = asset_record.groupby(['trade_year']).size().reset_index()
    yr_days.columns = ['trade_year', 'days']
    return_yr = pd.merge(return_yr, yr_days, on=['trade_year'], how='left')
    return_yr['asset_yr'] = return_yr['asset'] / return_yr['asset'].shift(1).fillna(1)
    return_yr['return_yr'] = round(return_yr['asset_yr'] ** (252 / return_yr['days']) - 1, 5)
    sharpe_yr = ((asset_record.groupby(['trade_year'])['return_rate'].mean()
                  / asset_record.groupby(['trade_year'])['return_rate'].std()) * np.sqrt(252)).reset_index()
    sharpe_yr.columns = ['trade_year', 'sharpe_yr']
    max_drawdown_yr = (asset_record.groupby(['trade_year'])['max_draw_down'].min()).reset_index()
    yr = pd.merge(return_yr[['trade_year', 'return_yr']], sharpe_yr, on=['trade_year'], how='left')
    yr = pd.merge(yr, max_drawdown_yr, on=['trade_year'], how='left')

    # 总体收益、夏普、回撤
    return_all = asset_record.tail(1)
    all_days = asset_record.shape[0]
    return_all['return_all'] = round(return_all['asset'] ** (252 / all_days) - 1, 5)
    sharpe_all = ((asset_record['return_rate'].mean()
                   / asset_record['return_rate'].std()) * np.sqrt(252))
    max_drawdown_all = (asset_record['max_draw_down'].min())
    all_record = pd.DataFrame({'return_all': [return_all['return_all'].iloc[0]], 'sharpe_all': sharpe_all,
                         'max_drawdown_all': max_drawdown_all})

    # 输出文档
    test_doc = Document()
    style = 'Light List Accent 2'
    default_section = test_doc.sections[0]
    default_section.page_width = Cm(30)  # 纸张大小改为自定义，方便放下大表和大图
    test_doc.styles['Normal'].font.name = 'Times New Roman'
    test_doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
    title = date_list[0].strftime('%Y-%m-%d') + '至' + date_list[-1].strftime('%Y-%m-%d') + '回测'
    add_heading(title, test_doc, level=0, seq=1)
    add_heading('整体业绩表现', test_doc, level=1, seq=1)
    all_record.columns = ['超额年化收益率', '超额夏普比率', '超额最大回撤']
    all_record['超额年化收益率'] = all_record.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    all_record['超额夏普比率'] = all_record.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    all_record['超额最大回撤'] = all_record.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)
    add_table(all_record.T.reset_index().T, test_doc, size=7, style=style, width=8)

    images = os.path.join(r'./figure_save/abs_asset.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/increase_asset.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/turnover_rate.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/holding_value.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/holding_index_weight.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/beta.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/momentum.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/size.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/earnyild.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/resvol.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/growth.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/btop.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/leverage.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/liquidty.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/sizenl.jpg')
    doc_add_images(test_doc, images, inch=6)
    add_heading('逐年业绩表现', test_doc, level=1, seq=2)
    num = 0  # 初始计数
    yr.columns = ['年份', '超额年化收益率', '超额夏普比率', '超额最大回撤']
    yr['年份'] = yr['年份'].astype('str')
    yr['超额年化收益率'] = yr.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    yr['超额夏普比率'] = yr.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    yr['超额最大回撤'] = yr.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)

    for year_ in asset_record['trade_year'].unique():
        num += 1
        add_heading('%s业绩表现'%year_, test_doc, level=2, seq=num)
        table_ = yr[yr['年份'] == str(year_)]
        add_table(table_.T.reset_index().T, test_doc, size=7, style=style, width=8)
    test_doc.save(r"./%s.docx"%file_name)


def backtest_single_index(score_df, turnover_punish, file_name):

    # 个股数据整理
    # 生成个股和指数涨跌幅
    stock_return = pd.read_pickle(os.path.join(jc.database_path, jc.stock_price_data))
    stock_return['open'] = stock_return['s_dq_open'] * stock_return['s_dq_adjclose'] / stock_return['s_dq_close']  # 后复权后开盘价
    stock_return['next_open'] = stock_return.groupby('security_code')['open'].shift(-1)
    stock_return['next_close'] = stock_return.groupby('security_code')['s_dq_adjclose'].shift(-1)
    stock_return['next_tvol'] = stock_return.groupby('security_code')['tvol'].shift(-1)
    chosen_index = jc.benchmark
    index_return = pd.read_pickle(os.path.join(jc.database_path, jc.index_price))
    index_return = index_return[index_return.security_code == chosen_index]

    # 预测值加入行业数据
    industry_cate = pd.read_pickle(os.path.join(jc.database_path, jc.industry_cate_data))
    industry_cate.drop_duplicates(inplace=True)
    daily_df = pd.merge(stock_return[['security_code', 'trade_date', 's_dq_close']], industry_cate,
                        on=['security_code', 'trade_date'], how='outer')
    daily_df.sort_values(by=['security_code', 'trade_date'], inplace=True)
    daily_df['sw_industry'] = daily_df.groupby('security_code')['sw_industry'].fillna(method='ffill')
    daily_df.dropna(inplace=True)  # 删除过早日期以及行业数据缺乏的数据
    # 预测值加入Barra数据
    barra = pd.read_pickle(os.path.join(jc.database_path, jc.barra_factor_data))
    daily_df = pd.merge(daily_df, barra, on=['trade_date', 'security_code'], how='left')
    daily_df.dropna(inplace=True)  # 删除一直停牌导致没有barra值的个股
    # 逐级拆分预期收益率
    init_date = score_df['trade_date'].min()
    end_date = score_df['trade_date'].max()
    score_df = pd.merge(left=score_df, right=daily_df, how='outer')
    score_df.rename(columns={list(score_df.filter(like='score_').columns)[0]: 'chg'}, inplace=True)
    score_df.fillna(-999, inplace=True)  #  不在选股池内，填负值
    score_df = score_df[(score_df['trade_date'] >= init_date) & (score_df['trade_date'] <= end_date)]
    # 指数数据整理
    # 生成指数成分股的barra和行业
    df_index = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    df_index = df_index[df_index.index_code == chosen_index]
    df_index['weight'] = df_index['weight'] / 100
    del df_index['index_code']
    index_weight = pd.merge(df_index[['trade_date']].drop_duplicates(), barra[['security_code', 'trade_date']],
                            on=['trade_date'], how='inner')  # 去权重公布日的所有个股数据，需要barra数据早于index权重数据
    index_weight = pd.merge(index_weight, df_index, on=['security_code', 'trade_date'], how='left')  # 合并入权重
    index_weight.fillna(0, inplace=True)
    index_weight = pd.merge(index_weight, barra, on=['security_code', 'trade_date'], how='outer')  # 合并入barra
    index_weight = pd.merge(index_weight, industry_cate, on=['security_code', 'trade_date'], how='outer')  # 合并入行业
    index_weight.sort_values(by=['security_code', 'trade_date'], inplace=True)
    index_weight['weight'] = index_weight.groupby('security_code')['weight'].fillna(method='ffill')
    index_weight['sw_industry'] = index_weight.groupby('security_code')['sw_industry'].fillna(method='ffill')
    index_weight.dropna(inplace=True)
    # 计算成分股的barra和行业占比
    barra_list = score_df.filter(like='barra_').columns.tolist()
    barra_std_list = [barra_ + '_std' for barra_ in barra_list]
    industry_list = list(score_df['sw_industry'].unique())
    df_index = pd.DataFrame(columns=barra_list + industry_list + barra_std_list)
    for date_, df_ in index_weight.groupby('trade_date'):
        df_ = df_[df_['weight'] != 0]
        df_[barra_list] = df_[barra_list].T.mul(list(df_['weight'])).T * len(df_)
        df_index.loc[date_, barra_list] = df_[barra_list].mean()  # 指数barra
        df_index.loc[date_, barra_std_list] = list(df_[barra_list].std())  # 指数barra标准差
        df_index.loc[date_, industry_list] = df_['sw_industry'].value_counts() / len(df_)
    df_index.fillna(0, inplace=True)

    # barra exporsure上下限
    for barra_ in barra_list:
        df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
        df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

    # 逐日回测（包含组合优化）
    date_list = score_df['trade_date'].drop_duplicates().sort_values().to_list()
    opt_weight = {}  # 字典形式储存结果
    initial_asset = 1000
    asset_record = pd.DataFrame(columns=['asset'])  # 纪录每日净值
    holding_record = {}  # 持仓权重，尾盘更新
    holding_record[date_list[0]] = pd.DataFrame({'security_code':['cash'],
                                                 'past_value':[initial_asset]}).set_index('security_code')
    turnover_record = pd.DataFrame(columns=['turnover_rate'])  # 纪录换手率
    holding_value_record = pd.DataFrame(columns=['num', 'value'])  # 记录持仓数量和平均市值
    # 记录Barra持仓
    barra_beta = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_momentum = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_size = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_earnyild = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_resvol = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_growth = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_btop = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_leverage = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_liquidty = pd.DataFrame(columns=['index', 'diff', 'barra'])
    barra_sizenl = pd.DataFrame(columns=['index', 'diff', 'barra'])
    # 读取市值
    stock_value = pd.read_pickle(os.path.join(jc.database_path, jc.mkt_capt_data))
    excel_df = []
    holding_index_weight = pd.DataFrame(columns=['300', '500', '1000', 'else'])
    stock_index_weight = pd.read_pickle(os.path.join(jc.database_path, jc.index_weight))
    index_stock_300 = stock_index_weight[stock_index_weight['index_code'] == '000300']
    index_stock_500 = stock_index_weight[stock_index_weight['index_code'] == '000905']
    index_stock_1000 = stock_index_weight[stock_index_weight['index_code'] == '000852']
    stock_index_date_300 = list(index_stock_300['trade_date'].unique())
    stock_index_date_500 = list(index_stock_500['trade_date'].unique())
    stock_index_date_1000 = list(index_stock_1000['trade_date'].unique())
    stock_index_date_300.sort()
    stock_index_date_500.sort()
    stock_index_date_1000.sort()
    index_stock_df = stock_index_weight[stock_index_weight['index_code'] == jc.benchmark]
    for date_num in range(0, len(date_list) - 1):
    # for date_num in range(0, 42):
        # date_ = pd.to_datetime('2022-12-30')
        date_ = date_list[date_num]
        next_date = date_list[date_num + 1]
        index_change_date = index_stock_df[index_stock_df['trade_date'] <= date_]['trade_date'].max()
        index_stock_today = index_stock_df[index_stock_df['trade_date'] == index_change_date]
        stock_today = score_df[score_df['trade_date'] == date_]
        stock_today.drop_duplicates(inplace=True)
        stock_today.loc[~stock_today['security_code'].isin(list(index_stock_today['security_code'])), 'chg'] = -999
        index_today = df_index.loc[date_]
        holding_df = holding_record[date_]
        past_weight = holding_df.reset_index()  # 根据净值计算权重
        past_weight['past_value'] = past_weight['past_value'] / past_weight['past_value'].sum()
        past_weight.columns = ['security_code', 'past_weight']
        past_weight = past_weight[past_weight['security_code'] != 'cash']
        past_weight = pd.merge(left=stock_today[['security_code']], right=past_weight,
                               on='security_code', how='outer')
        past_weight.fillna(0, inplace=True)
        stock_today = pd.merge(left=stock_today, right=past_weight[['security_code']],
                               on='security_code', how='outer')
        stock_today['trade_date'] = date_

        # 计算当日opt_weight, 无解情况下则放宽限制
        barra_std_num = mc.barra_std_num
        try:
            opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                 stock_today, index_today, turnover_punish)
        except:
            barra_std_num += mc.barra_std_num / 2
            for barra_ in barra_list:
                df_index.loc[date_, barra_ + '_lower'] = \
                (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
                df_index.loc[date_, barra_ + '_upper'] = \
                (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
            index_today = df_index.loc[date_]
            opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                 stock_today, index_today, turnover_punish)

        # trade_target = pd.DataFrame(opt_weight[date_].values(), index=opt_weight[date_].keys()).reset_index()
        # trade_target.columns = ['security_code', 'weight']
        # trade_target['trade_date'] = date_
        # today_barra = pd.merge(left=trade_target, right=stock_today, on=['security_code', 'trade_date'], how='inner')
        # today_barra = pd.merge(left=trade_target, right=barra, on=['security_code', 'trade_date'], how='inner')

        while status not in ['optimal', 'optimal_inaccurate']:
            barra_std_num += mc.barra_std_num / 2
            for barra_ in barra_list:
                df_index.loc[date_, barra_ + '_lower'] = (df_index[barra_] - barra_std_num * df_index[barra_ + '_std']).loc[date_]
                df_index.loc[date_, barra_ + '_upper'] = (df_index[barra_] + barra_std_num * df_index[barra_ + '_std']).loc[date_]
            index_today = df_index.loc[date_]
            opt_weight[date_], status = port_opt(['chg'], industry_list, barra_list, past_weight,
                                                 stock_today, index_today, turnover_punish)
        # 进行交易
        trade_target = pd.DataFrame(opt_weight[date_].values(), index=opt_weight[date_].keys()).reset_index()
        trade_target.columns = ['security_code', 'trade_weight']
        today_stock_return = stock_return[stock_return['trade_date'] == date_]
        holding_df, turnover_rate = trade_cal(past_weight, holding_df, trade_target, today_stock_return)
        # 计算成分股比例
        holding_index = holding_df.copy()
        holding_index['weight'] = holding_index['past_value'] / holding_index['past_value'].sum()
        weight_300 = 0
        weight_500 = 0
        weight_1000 = 0
        weight_else = 0
        last_300_date = [date_ for date_ in stock_index_date_300 if date_ <= next_date][-1]
        last_500_date = [date_ for date_ in stock_index_date_500 if date_ <= next_date][-1]
        last_1000_date = [date_ for date_ in stock_index_date_1000 if date_ <= next_date][-1]
        index_stock_300_today = index_stock_300[index_stock_300['trade_date'] == last_300_date]
        index_stock_500_today = index_stock_500[index_stock_500['trade_date'] == last_500_date]
        index_stock_1000_today = index_stock_1000[index_stock_1000['trade_date'] == last_1000_date]
        for stock_ in holding_index.index:
            if stock_ in list(index_stock_300_today['security_code']):
                weight_300 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_500_today['security_code']):
                weight_500 += holding_index.loc[stock_, 'weight']
            elif stock_ in list(index_stock_1000_today['security_code']):
                weight_1000 += holding_index.loc[stock_, 'weight']
            else:
                weight_else += holding_index.loc[stock_, 'weight']
        holding_index_weight.loc[next_date] = [weight_300, weight_500, weight_1000, weight_else]
        # 净值和持仓记录
        holding_record[next_date] = holding_df
        asset_record.loc[next_date] = holding_df['past_value'].sum() / initial_asset
        turnover_record.loc[next_date] = turnover_rate
        today_mkt = pd.DataFrame(holding_df).reset_index()
        today_mkt['trade_date'] = next_date
        today_stock_capt = stock_value[stock_value['trade_date'] == next_date][['security_code',
                                                                                'trade_date', 'total_capt']]
        today_mkt = pd.merge(left=today_mkt, right=today_stock_capt, on=['security_code', 'trade_date'], how='inner')
        # mkt_value = (today_mkt['free_capt'] * today_mkt['past_value'] / today_mkt['past_value'].sum()).sum() / 10000
        mkt_value = today_mkt['total_capt'].median() / 10000
        holding_value_record.loc[next_date] = [len(today_mkt), mkt_value]
        excel_df.append(today_mkt)
        # 计算barra暴露
        today_barra = pd.DataFrame(holding_df).reset_index()
        today_barra['trade_date'] = next_date
        today_barra_all = barra[barra['trade_date'] == next_date]
        today_barra = pd.merge(left=today_barra, right=today_barra_all, on=['security_code', 'trade_date'], how='inner')
        today_barra[barra_list] = today_barra[barra_list] * today_barra['past_value'].values.reshape(-1, 1) \
                                  / today_mkt['past_value'].sum()
        index_next = df_index.loc[next_date]
        diff = (today_barra['barra_beta'].sum() - index_next['barra_beta']) / (
                    (index_next['barra_beta_upper'] - index_next['barra_beta']) / mc.barra_std_num)
        barra_beta.loc[next_date] = [index_next['barra_beta'], diff, today_barra['barra_beta'].sum()]
        diff = (today_barra['barra_momentum'].sum() - index_next['barra_momentum']) / (
                (index_next['barra_momentum_upper'] - index_next['barra_momentum']) / mc.barra_std_num)
        barra_momentum.loc[next_date] = [index_next['barra_momentum'], diff, today_barra['barra_momentum'].sum()]
        diff = (today_barra['barra_size'].sum() - index_next['barra_size']) / (
                (index_next['barra_size_upper'] - index_next['barra_size']) / mc.barra_std_num)
        barra_size.loc[next_date] = [index_next['barra_size'], diff, today_barra['barra_size'].sum()]
        diff = (today_barra['barra_earnyild'].sum() - index_next['barra_earnyild']) / (
                (index_next['barra_earnyild_upper'] - index_next['barra_earnyild']) / mc.barra_std_num)
        barra_earnyild.loc[next_date] = [index_next['barra_earnyild'], diff, today_barra['barra_earnyild'].sum()]
        diff = (today_barra['barra_resvol'].sum() - index_next['barra_resvol']) / (
                (index_next['barra_resvol_upper'] - index_next['barra_resvol']) / mc.barra_std_num)
        barra_resvol.loc[next_date] = [index_next['barra_resvol'], diff, today_barra['barra_resvol'].sum()]
        diff = (today_barra['barra_growth'].sum() - index_next['barra_growth']) / (
                (index_next['barra_growth_upper'] - index_next['barra_growth']) / mc.barra_std_num)
        barra_growth.loc[next_date] = [index_next['barra_growth'], diff, today_barra['barra_growth'].sum()]
        diff = (today_barra['barra_btop'].sum() - index_next['barra_btop']) / (
                (index_next['barra_btop_upper'] - index_next['barra_btop']) / mc.barra_std_num)
        barra_btop.loc[next_date] = [index_next['barra_btop'], diff, today_barra['barra_btop'].sum()]
        diff = (today_barra['barra_leverage'].sum() - index_next['barra_leverage']) / (
                (index_next['barra_leverage_upper'] - index_next['barra_leverage']) / mc.barra_std_num)
        barra_leverage.loc[next_date] = [index_next['barra_leverage'], diff, today_barra['barra_leverage'].sum()]
        diff = (today_barra['barra_liquidty'].sum() - index_next['barra_liquidty']) / (
                (index_next['barra_liquidty_upper'] - index_next['barra_liquidty']) / mc.barra_std_num)
        barra_liquidty.loc[next_date] = [index_next['barra_liquidty'], diff, today_barra['barra_liquidty'].sum()]
        diff = (today_barra['barra_sizenl'].sum() - index_next['barra_sizenl']) / (
                (index_next['barra_sizenl_upper'] - index_next['barra_sizenl']) / mc.barra_std_num)
        barra_sizenl.loc[next_date] = [index_next['barra_sizenl'], diff, today_barra['barra_sizenl'].sum()]
    # excel_df = pd.concat(excel_df, axis=0)
    # excel_df.to_excel('回测监控.xlsx')
    # 生成回测文档
    asset_record.loc[asset_record.index[0] - timedelta(days=1), 'asset'] = 1  # 加入初始日期
    asset_record.sort_index(inplace=True)
    asset_record.reset_index(inplace=True)
    asset_record.columns = ['trade_date', 'origin']
    index_init = index_return[index_return['trade_date'] <= asset_record['trade_date'].iloc[0]]['close'].iloc[-1]
    asset_record = pd.merge(left=asset_record, right=index_return[['trade_date', 'close']], on=['trade_date'], how='left')
    asset_record.fillna(index_init, inplace=True)  # 有可能指数在首日没有数值
    asset_record['close'] = asset_record['close'] / index_init
    asset_record['asset'] = asset_record['origin'] / asset_record['close']

    asset_record['return_rate'] = asset_record['asset'].pct_change()
    asset_record['max_draw_down'] = asset_record['asset'].expanding().max()  # 累计收益率的历史最大值
    asset_record['max_draw_down'] = asset_record['asset'] / asset_record['max_draw_down'] - 1  # 回撤
    asset_record['trade_year'] = asset_record['trade_date'].dt.year

    # 图1：绝对收益、指数收益、超额收益
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为1天
    ax2.bar(list(asset_record['trade_date']), list(asset_record['asset'] - 1), width=bar_width,
            color='skyblue', label='累计超额收益', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['origin']), color='red', label='策略绝对净值')
    ax1.plot(list(asset_record['trade_date']), list(asset_record['close']), color='black', label='指数净值')
    plt.title('绝对收益')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r"./figure_save/abs_asset.jpg", dpi=1000)

    # 图2： 超额收益、超额最大回撤
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    ax2 = ax1.twinx()
    bar_width = pd.Timedelta(days=1)  # 设置柱状图柱状的宽度为xx天 len(date_list)/50+1
    ax2.bar(asset_record['trade_date'], asset_record['max_draw_down'], width=bar_width,
            color='skyblue', label='最大回撤', alpha=0.5)
    ax1.plot(list(asset_record['trade_date']), list(asset_record['asset']), color='red', label='净值走势')
    plt.title('超额收益及超额最大回撤')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.savefig(r"./figure_save/increase_asset.jpg", dpi=1000)
    # 图3：换手率走势
    fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    turnover_figure = turnover_record.iloc[1:]  # 第一天建仓，数据没意义
    ax1.plot(list(turnover_figure.index), list(turnover_figure['turnover_rate']), color='red', label='换手率走势')
    plt.title('换手率')
    lines, labels = ax1.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/turnover_rate.jpg", dpi=1000)
    # 图4：平均市值变动
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(list(holding_value_record.index), list(holding_value_record['num']), color='red', label='持仓个股数目')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('持仓个股数目', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(list(holding_value_record.index), list(holding_value_record['value']), color='blue', label='总市值中位数')
    ax2.set_ylabel('总市值中位数', color='blue')
    ax2.tick_params('y', colors='blue')
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines])
    plt.title('持仓数量和市值')
    plt.savefig(r"./figure_save/holding_value.jpg", dpi=1000)
    # 图5：持仓股比例
    plt.figure(figsize=(10, 6))
    ax = holding_index_weight.plot(kind='bar', stacked=True)
    plt.xticks(range(len(holding_index_weight.index)), holding_index_weight.index.strftime('%Y-%m-%d'), rotation=30)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('指数成分股占比')
    plt.show()
    plt.savefig(r"./figure_save/holding_index_weight.jpg", dpi=1000)
    # 图5-15：barra图
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_beta
    name = 'beta'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势'%name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限'%name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限'%name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离'%name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势'%name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg"%name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_momentum
    name = 'momentum'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_size
    name = 'size'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_earnyild
    name = 'earnyild'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_resvol
    name = 'resvol'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_growth
    name = 'growth'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_btop
    name = 'btop'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_leverage
    name = 'leverage'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_liquidty
    name = 'liquidty'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    fig, ax = plt.subplots(figsize=(10, 5))  # 创建一个图表对象fig和一个坐标轴对象ax1，设置画布大小
    df = barra_sizenl
    name = 'sizenl'
    ax.plot(list(df.index), list(df['index']), color='red', label='指数%s走势' % name)
    ax2 = ax.twinx()
    ax2.plot(list(df.index), [mc.barra_std_num] * len(df.index), color='blue', label='%s上限' % name)
    ax2.plot(list(df.index), [-mc.barra_std_num] * len(df.index), color='blue', label='%s下限' % name)
    ax2.plot(list(df.index), list(df['diff']), color='y', label='策略%s隔日偏离' % name)
    ax2.set_ylim(- 3 * mc.barra_std_num, 3 * mc.barra_std_num)
    plt.title('%s走势' % name)
    lines, labels = ax.get_legend_handles_labels()
    plt.legend(lines, labels, loc='upper left')
    plt.savefig(r"./figure_save/%s.jpg" % name, dpi=1000)

    # 逐年收益、夏普、回撤
    return_yr = asset_record.groupby(['trade_year']).last().reset_index()
    yr_days = asset_record.groupby(['trade_year']).size().reset_index()
    yr_days.columns = ['trade_year', 'days']
    return_yr = pd.merge(return_yr, yr_days, on=['trade_year'], how='left')
    return_yr['asset_yr'] = return_yr['asset'] / return_yr['asset'].shift(1).fillna(1)
    return_yr['return_yr'] = round(return_yr['asset_yr'] ** (252 / return_yr['days']) - 1, 5)
    sharpe_yr = ((asset_record.groupby(['trade_year'])['return_rate'].mean()
                  / asset_record.groupby(['trade_year'])['return_rate'].std()) * np.sqrt(252)).reset_index()
    sharpe_yr.columns = ['trade_year', 'sharpe_yr']
    max_drawdown_yr = (asset_record.groupby(['trade_year'])['max_draw_down'].min()).reset_index()
    yr = pd.merge(return_yr[['trade_year', 'return_yr']], sharpe_yr, on=['trade_year'], how='left')
    yr = pd.merge(yr, max_drawdown_yr, on=['trade_year'], how='left')

    # 总体收益、夏普、回撤
    return_all = asset_record.tail(1)
    all_days = asset_record.shape[0]
    return_all['return_all'] = round(return_all['asset'] ** (252 / all_days) - 1, 5)
    sharpe_all = ((asset_record['return_rate'].mean()
                   / asset_record['return_rate'].std()) * np.sqrt(252))
    max_drawdown_all = (asset_record['max_draw_down'].min())
    all_record = pd.DataFrame({'return_all': [return_all['return_all'].iloc[0]], 'sharpe_all': sharpe_all,
                         'max_drawdown_all': max_drawdown_all})

    # 输出文档
    test_doc = Document()
    style = 'Light List Accent 2'
    default_section = test_doc.sections[0]
    default_section.page_width = Cm(30)  # 纸张大小改为自定义，方便放下大表和大图
    test_doc.styles['Normal'].font.name = 'Times New Roman'
    test_doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
    title = date_list[0].strftime('%Y-%m-%d') + '至' + date_list[-1].strftime('%Y-%m-%d') + '回测'
    add_heading(title, test_doc, level=0, seq=1)
    add_heading('整体业绩表现', test_doc, level=1, seq=1)
    all_record.columns = ['超额年化收益率', '超额夏普比率', '超额最大回撤']
    all_record['超额年化收益率'] = all_record.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    all_record['超额夏普比率'] = all_record.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    all_record['超额最大回撤'] = all_record.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)
    add_table(all_record.T.reset_index().T, test_doc, size=7, style=style, width=8)

    images = os.path.join(r'./figure_save/abs_asset.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/increase_asset.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/turnover_rate.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/holding_value.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/holding_index_weight.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/beta.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/momentum.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/size.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/earnyild.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/resvol.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/growth.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/btop.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/leverage.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/liquidty.jpg')
    doc_add_images(test_doc, images, inch=6)
    images = os.path.join(r'./figure_save/sizenl.jpg')
    doc_add_images(test_doc, images, inch=6)
    add_heading('逐年业绩表现', test_doc, level=1, seq=2)
    num = 0  # 初始计数
    yr.columns = ['年份', '超额年化收益率', '超额夏普比率', '超额最大回撤']
    yr['年份'] = yr['年份'].astype('str')
    yr['超额年化收益率'] = yr.apply(lambda x: str(round(x['超额年化收益率'] * 100, 2)) + '%', axis=1)
    yr['超额夏普比率'] = yr.apply(lambda x: str(round(x['超额夏普比率'], 2)), axis=1)
    yr['超额最大回撤'] = yr.apply(lambda x: str(round(x['超额最大回撤'] * 100, 2)) + '%', axis=1)

    for year_ in asset_record['trade_year'].unique():
        num += 1
        add_heading('%s业绩表现'%year_, test_doc, level=2, seq=num)
        table_ = yr[yr['年份'] == str(year_)]
        add_table(table_.T.reset_index().T, test_doc, size=7, style=style, width=8)
    test_doc.save(r"./%s.docx"%file_name)


