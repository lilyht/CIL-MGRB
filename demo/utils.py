import numpy as np
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors

wv_from_bin = KeyedVectors.load_word2vec_format("./save/google.bin", binary=True)  # C bin format

########################################
# Add these functions on the original file 'utils.py'
def get_structure(args):
    '''
    Args:
        args
    Retruns:
        test_num_lst, classes, datainfopath, nodepathinfo
    '''
    mini_nodepathinfo_path = '../data/mini-imagenet/processed/nodepathinfo.npy'
    i100_nodepathinfo_path = '../data/imagenet100_resized_256/data/nodepathinfo.npy'
    if args["dataset"] == 'cifar100':
        classes = list(range(0, 100))
        datainfopath = './save/data_divide_info_{}_{}.npy'.format(args["initial_increment"], args["increment"])
        # datainfopath = './save/data_divide_info.npy'
        nodepathinfo = np.load(mini_nodepathinfo_path, allow_pickle=True) # cifar数据集没有用到
    elif args["dataset"] == 'cifar10':
        classes = list(range(0, 10))
        datainfopath = './save/cifar10_divide_info_{}_{}.npy'.format(args["initial_increment"], args["increment"])
        # datainfopath = './save/cifar10_divide_info.npy'
        nodepathinfo = np.load(mini_nodepathinfo_path, allow_pickle=True)
    elif args["dataset"] == 'miniimagenet':
        classes = list(range(0, 100))
        datainfopath = './save/miniimagenet_divide_info_{}_{}.npy'.format(args["initial_increment"], args["increment"])
        nodepathinfo = np.load(mini_nodepathinfo_path, allow_pickle=True)
    elif args["dataset"] == 'imagenet100':
        classes = list(range(0, 100))
        datainfopath = './save/imagenet100_divide_info_{}_{}.npy'.format(args["initial_increment"], args["increment"])
        nodepathinfo = np.load(i100_nodepathinfo_path, allow_pickle=True)
    if args["cluster"] == -2:
        print("using WordNet tree structure")  # The knowledge structure grows gradually
    elif args["cluster"] == -1:
        print("using original tree structure(for cifar100)")
    else:
        print("using word embedding to generate tree structure({} clusters)".format(args["cluster"]))
    datainfo = np.load(datainfopath, allow_pickle=True)
    test_num_lst = datainfo.item()["test_num"]
    return test_num_lst, classes, datainfopath, nodepathinfo

def get_coarse_label(args, order_lst, test_num_lst, inc_idx):
    '''
    Args:
        order_lst(list): the learning order with seed=1993
        test_num_lst(list): The number of classes to be learnt in every phase. Generate and store for convenience.
                            For example, test_num_lst = [50, 60, 70, 80, 90, 100]
        inc_idx(int): number of the current phase (strat from 0)
    Returns:
        coarse_labe(list)
        max_depth(int): max depth of the tree structure
    '''
    
    if args["dataset"] == "cifar10":
        coarse_label = semantic_hier(args, order_lst, test_num_lst[inc_idx], 2)
        max_depth = 2
    elif "hierloss" in args["setloss"] and args["dataset"] == "imagenet100":
        print("### hierloss & imagenet ###")
        coarse_label = [0]
        max_depth = cal_height(args, test_num_lst[inc_idx])
    else:
        coarse_label = [0]
        max_depth = 2
    return coarse_label, max_depth

def getCluster(ds, labellst, cur_clsnum, n_clusters):
    '''
    KMeans
    Args: 
        ds(str): dataset
        labellst(list): the randlst of the dataset, defined in dividecifar100.py
        cur_clsnum(int): number of appearing classes currently
        n_clusters(int): number of clusters
    Returns:
        coarse_label(list): the coarse label of classes from original class 0 
        to class cur_clsnum
    '''
    label_names = []
    if ds == 'cifar10':
        label_names = cifar10_label_names
    elif ds == 'cifar100':
        label_names = cifar100_label_names
    elif ds == 'imagenet':
        label_names = mini_label_names
    
    # print("Current clustering: {} classes in total".format(cur_clsnum))
    # print("labellst:", labellst)

    veclst = []
    coarse_label = []
    tmp = []
    for i in range(cur_clsnum):
        # print("labellst[i]:", labellst[i])
        tmp.append(label_names[labellst[i]])
        # print(label_names[labellst[i]], end=" ")
        vec = wv_from_bin[label_names[labellst[i]]]
        veclst.append(vec)
    X = np.array(veclst)
    if cur_clsnum < n_clusters:  # 当cur_clsnum太小时
        n_clusters = 1
    kmeans = KMeans(n_clusters, random_state=0).fit(X)
    for idx in range(cur_clsnum):
        coarse_label.append(kmeans.labels_[idx])
    return coarse_label

def semantic_hier(args, labellst, curclsnum, cluster_num=2):
    '''    
    Args:
        args
        labellst(list): the randlst of the dataset, defined in dividecifar100.py
        cur_clsnum(int): number of appearing claases currently
        cluster_num(int): number of clusters
    Returns:
        coarse_label(list)
    '''
    coarse_label = getCluster(args["dataset"], labellst, curclsnum, cluster_num)
    return coarse_label

def cal_height(args, cur_clsnum):
    '''
    Args:
        args:
        cur_clsnum(int): number of appearing claases currently
    Returns:
        height(int): the height of the LCS Tree
    '''
    test_num_lst, classes, datainfopath, nodepathinfo = get_structure(args)
    clsid_path = nodepathinfo.item()["clsid_path"]

    path_collect = []
    minlen = 18
    maxlen = 0
    for i in range(cur_clsnum):
        path_i = clsid_path[i]
        path_collect.append(path_i)
        if minlen > len(path_i):
            minlen = len(path_i)
        if maxlen < len(path_i):
            maxlen = len(path_i)
    common = 1
    for i in range(0, minlen):
        flag = True
        cur_node = path_collect[0][i]
        for j in range(cur_clsnum):
            if path_collect[j][i] != cur_node:
                flag = False
                break
        if flag == False:
            break
        else:
            common = common + 1
    height = maxlen - common
    print("height:", height)
    return height


cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']

cifar100_label_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

cifar100_label_names_sort = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 
    'large_man-made_outdoor_things', 'large_natural_outdoor scenes', 'large_omnivores_and_herbivores',
    'medium-sized_mammals', 'non-insect_invertebrates', 'people',
    'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2', 
    'beaver', 'dolphin', 'otter', 'seal', 'whale', 
    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    'bottle', 'bowl', 'can', 'cup', 'plate',
    'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
    'clock', 'keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
]

mini_label_names = [
    'fish', 'goose', 'missile', 'cabinet', 'bars', 'dog', 'clog', 'carousel', 'poncho', 'curtain',
    'ladybug', 'snorkel', 'ferret', 'holster', 'dustcart', 'frying_pan', 'reel', 'Tibetan_mastiff', 'consomme', 'breastplate',
    'container', 'spider_web', 'chime', 'dog', 'dog', 'crate', 'miniskirt', 'trifle', 'barrel', 'stage',
    'dog', 'oboe', 'bookshop', 'dalmatian', 'dugong', 'bus', 'orange', 'slot', 'miniature_poodle', 'lock',
    'bolete', 'jellyfish', 'bar', 'vase', 'ashcan', 'golden_retriever', 'dome', 'iPod', 'tank', 'carton',
    'dog', 'daddy_longlegs', 'ant', 'fox', 'dog', 'dog', 'finch', 'scoreboard', 'spike', 'wok',
    'photocopier', 'robin', 'accessories', 'rhinoceros_beetle', 'furnace', 'boxer', 'catamaran', 'bottle', 'dog', 'dishrag',
    'triceratops', 'unicycle', 'lipstick', 'fence', 'electric_guitar', 'yawl', 'sloth', 'nematode', 'pipe_organ', 'malamute',
    'tobacconist_shop', 'hourglass', 'king_crab', 'toucan', 'meerkat', 'cliff', 'sign', 'roof', 'rug', 'green_mamba',
    'wolf', 'lion', 'upright_piano', 'coral_reef', 'hotdog', 'aircraft_carrier', 'fireguard', 'cocktail_shaker', 'cannon', 'mixing_bowl']

# End
########################################