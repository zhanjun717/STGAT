import glob

'''
Reference https://github.com/d-ailin/GDN
'''

def get_feature_map(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list

# graph is 'fully-connect'
def get_fc_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list: # 遍历所有特征
        if ft not in struc_map:  # 构建结构映射字典struc_map
            struc_map[ft] = []

        for other_ft in feature_list:   # 遍历所有特征
            if other_ft is not ft:      # 将除当前节点以外的所有节点保存到struc_map中，struc_map字典保存所有节点与其他节点的连接关系
                struc_map[ft].append(other_ft)
    
    return struc_map

def get_tc_graph_struc(temporal_len):
    struc_map = {}
    teporal_list = list(range(0,temporal_len))

    for tp in teporal_list:  # 遍历所有特征
        if tp not in struc_map:  # 构建结构映射字典struc_map
            struc_map[tp] = []

        for other_tp in teporal_list:  # 遍历所有特征
            if other_tp is not tp and other_tp < tp:  # 将除当前节点以外的所有节点保存到struc_map中，struc_map字典保存所有节点与其他节点的连接关系
                struc_map[tp].append(other_tp)

    return struc_map

def get_prior_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    return struc_map

if __name__ == '__main__':
    get_fc_graph_struc('SWat')