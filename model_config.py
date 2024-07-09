T_list = [1, 3, 5, 10, 22]  # 模型预测频率
# T_list = [1]
T_pre_list = [5, 10, 22]  # 模型预测频率
T_pre_list_300 = [10, 22]  # 300模型预测频率
# T_pre_list = [5]  # 模型预测频率
adjustment_frequency = 'monthly'  # 模型更新频率，可填monthly 或者weekly 或者quarter

# model_list = ['Xgb']
model_list = ['MLP', 'Xgb']
loss_cate = 'WCCC'
train_window_dict = {'3': 365*2,
                     '5': 365*2,
                     '10': 365*2,
                     '22': 365*2}  # 训练窗口日期

# 组合优化参数
trade_cost = 0.001  # 千1滑点
buy_fee = 0.0089 / 100  # 买的手续费
sell_fee = 0.0589 / 100  # 卖的手续费
industry_diff = 0.02  # 行业偏离
barra_std_num = 0.5
size_thresh = 0.1
turnover_punish = 500  # 换手率的惩罚项
risk_aversion = 100 # 风险厌恶系数

# industry_diff = 100  # 行业偏离
# barra_std_num = 100
# turnover_punish = 2  # 换手率的惩罚项