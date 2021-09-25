# preprocess data
import numpy as np
import re

def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0):
    res = []

    # 判断数据中是否存在所有的节点，如果特征不存在则丢弃这条数据
    for feature in feature_map: # 遍历所有特征节点
        if feature in data.columns: # 如果特征存在数据中
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])  # 返回有效数据的条数

    if type(labels) == int: # 如果数据中不含标签，则数据标签位全部填0
        res.append([labels]*sample_n)
    elif len(labels) == sample_n: # 如果数据中含标签，则将标签追加到数据中
        res.append(labels)

    return res

# 此函数将所有节点与其子集节点的对应字典，转换成节点编号对应的边连接关系矩阵
def build_loc_net(struc, all_features, feature_map=[]):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)   # 获取节点名称所在的编号
        for child in node_list:                        # 遍历不包含当前节点的其他子节点
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)    # 获取子节点名称所在的编号
            edge_indexes[0].append(c_index)             # 保存当前子节点的编号
            edge_indexes[1].append(p_index)             # 保存当前父节点的编号

    return edge_indexes