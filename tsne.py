import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np

import random

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
# colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
#           'hotpink']

colors = plt.cm.get_cmap('tab10')

# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, marker='o'):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]
    

    # class_index = random.sample(range(60), 5)
    # class_index.extend(new_class_index)
    class_index = [0, 1, 2, 3, 4]

    print(class_index)

    for i, index in enumerate(class_index):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']

        if marker == 'o':
            plt.scatter(X, Y, s=20, color=colors(i), label=i, marker=marker)
        else:
            plt.scatter(X, Y, s=150, color=colors(i), marker=marker)
        
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    # plt.title(name, fontsize=32, fontweight='normal', pad=20)

 # 每个特征的维度为512
feat = np.load('query.npy')
proto = np.load('proto.npy')
# label_test = np.load('save_array/clip_emb_label.npy')
label_query = np.array([0, 1, 2, 3, 4] * 30)
label_proto = np.array([0, 1, 2, 3, 4])


print(label_query)
print(label_query.shape)

# plt.figure(figsize=(10, 10))

visual_feat = visual(feat)
visual_proto = visual(proto)

plotlabels(visual_proto, label_proto, marker='*')

plotlabels(visual_feat, label_query)
plt.legend(loc='lower right')

plt.savefig('UESD.svg', format='svg', bbox_inches='tight', pad_inches=0.1)