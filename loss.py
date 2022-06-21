import utils
import cal_MG
import numpy as np

def cal_hier_loss(args, inc_idx, order_lst, outputs, targets):
    # 计算多粒度正则项的步骤：
    '''
    Args:
        args
        inc_idx(int): 
        order_lst(list) : a random list from 0-99, with seed=1993
        
    Return:
        hier_loss
    '''
    test_num_lst, classes, datainfopath, nodepathinfo = utils.get_structure(args)
    datainfo = np.load(datainfopath, allow_pickle=True)
    # 生成多粒度标签
    coarse_label, max_depth = utils.get_coarse_label(args, order_lst, test_num_lst, inc_idx)
    # 调用cal_mg.py计算MGloss
    hier_loss = cal_MG.MGloss(args, outputs["logits"], targets, "KL", inc_idx, datainfo,
                nodepathinfo, test_num_lst, coarse_label, args["mg_beta"], max_depth) * args["mg_scale"]
    return hier_loss