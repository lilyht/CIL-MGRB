import os
import numpy as np
from nltk.corpus import wordnet as wn

imagenet100_path = '../data/imagenet100_resized_256/data'

def del_repeated_file(save_path):
    if(os.path.exists(save_path) == True):
        os.remove(save_path)

# 生成并处理 i100_wordlist_tmp.txt
wnids_path = "./wnids.txt"
words_path = "./words.txt"
save_path = './i100_wordlist_tmp.txt'
del_repeated_file(save_path)

with open(wnids_path) as txt:
    for line in txt:
        label = line.strip('\n')
        with open(words_path) as dic:
            for dic_line in dic:
                idx_1 = dic_line.strip('\n').split('\t')[0]
                idx_2 = dic_line.strip('\n').split('\t')[1]
                if idx_1 == label:
                    with open(save_path, "a") as file:
                        file.write(str(idx_2)+"\n")
print("Generate i100_wordlist_tmp!")


# 处理 i100_wordlist_tmp.txt，删除每行逗号后的文字
load_path = './i100_wordlist_tmp.txt'
save_path = './i100_wordlist.txt'
del_repeated_file(save_path)

with open(load_path) as txt:
    for line in txt:
        label_1 = line.strip('\n').split(',')[0]
        with open(save_path, "a") as file:
            file.write(str(label_1)+"\n")
print("Generate i100_wordlist!")


# 获取上位词直到顶端

# 构建词典
with open(imagenet100_path + '/words.txt') as words:
    netdict = {}
    for line in words:
        netdict[line.strip('\n').split('\t')[1]] = line.strip('\n').split('\t')[0]

with open(imagenet100_path + '/words.txt') as words:
    netdictmore = {}
    for line in words:
        for sub in line.strip('\n').split('\t')[1].split(", "):
            netdictmore[sub] = line.strip('\n').split('\t')[0]
print("dicts have been built")


# 获得全部label
all_labels = []
all_words = []
with open(imagenet100_path + '/wnids.txt') as wnid:
    print("read wnids...")
    for line in wnid:
        all_labels.append(line.strip('\n'))
    print(len(all_labels))  #一共有100个分类
    
# 获得全部label对应的单词
with open(imagenet100_path + '/i100_wordlist.txt') as wordlist:
    print("read wordlist...")
    for line in wordlist:
        all_words.append(line.strip('\n'))
    print(len(all_words))

nodepath = {}
clsname2path = {}
clsid2path = {}

# -------------------------------------
def getnetnum(name):
    name = name.replace("_", " ")
    # print(name)
    if name in netdict:
        return netdict[name]
    else:
        return netdictmore[name]

def getnethier_cycle(i):
    path = []
    itself = all_labels[i]
    path.append(itself)
    
    name = all_words[i]
    cat = str(wn.synsets(name)[0]).split("'")[1]    # 获取多个词义，默认取第一个词义
    
    special_list = [1, 25, 35, 43, 44, 62, 63]  # 含义不对应的词，特殊标记并在下方进行更正
    print(cat)
    if i in special_list:
        cat = name + '.n.02'
    elif i == 32:
        cat = "binder.n.03"
    elif i == 56:
        cat = "hip.n.05"
    elif i == 69:
        cat = "cock.n.04"
    elif i == 77:
        cat = "chow.n.03"
    
    cat = wn.synset(cat).hypernyms()  #　【cat变量】的上位词
    
    if(cat == []):
        print("fail")
    else:
        while cat != []:
            catstr = str(cat)
            catstr = catstr.split("'")[1]
            path.append(getnetnum(catstr.split('.')[0]))
            cat = wn.synset(catstr).hypernyms()
            
    path = path[::-1]
    clsname2path[name] = path  # "xxx": ['nxxx', 'nxxx']
    clsid2path[i] = path  # 0: ['nxxx', 'nxxx'], 1: ['nxxx', 'nxxx']
    print("nodepath['{}']: {}".format(name, path))
    print("nodepath['{}']: {}".format(i, path))
    return len(path)
    
nodepath["clsname_path"] = clsname2path
nodepath["clsid_path"] = clsid2path

maxdepth = 0
for i in range(100):
    depth = getnethier_cycle(i)
    if maxdepth < depth:
        maxdepth = depth
        
print("maxdepth: {}".format(maxdepth))    
np.save('./nodepathinfo.npy', nodepath)
print("Finish!")