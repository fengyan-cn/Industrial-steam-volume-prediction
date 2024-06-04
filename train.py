import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

train_data_file = "zhengqi_train.txt"
test_data_file = "zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

column = train_data.columns.tolist()[:39]
# fig = plt.figure(figsize=(80, 60), dpi=75)
# for i in range(38):
#     plt.subplot(7, 8, i+1)
#     sns.boxplot(train_data[column[i]], orient='v', width=0.5)
#     plt.ylabel(column[i], fontsize=36)
# plt.show()

# find outliers

# train_cols = 6
# train_rows = len(train_data.columns)
# plt.figure(figsize=(4*train_cols, 4*train_rows))
# i=0
# for col in train_data.columns:
#     i += 1
#     ax = plt.subplot(train_rows, train_cols, i)
#     sns.distplot(train_data[col], fit=stats.norm)
# plt.tight_layout()
# plt.show()

# train_cols = 6
# train_rows = len(train_data.columns)
# plt.figure(figsize=(4*train_cols, 4*train_rows))
# i = 0
# for col in test_data.columns:
#     i += 1
#     ax = plt.subplot(train_rows, train_cols, i)
#     ax = sns.kdeplot(train_data[col], color="r", shade=True)
#     ax = sns.kdeplot(test_data[col], color="g", shade=True)
#     ax.set_xlabel(col)
#     ax.set_ylabel("freq")
#     ax = ax.legend(["train", "test"])
# plt.tight_layout()
# plt.show()


# regplot

# 计算相关性系数


