import os
import re
import numpy as np
from nltk.corpus import wordnet as wn

imagenet100_path = './data'
wnids_path = "./data/wnids.txt"
words_path = "./data/words.txt"
wlst_tmp_path = './data/i100_wordlist_tmp.txt'
wlst_path = './data/i100_wordlist.txt'

def del_repeated_file(path):
    if(os.path.exists(path) == True):
        os.remove(path)

# 生成并处理 i100_wordlist_tmp.txt
del_repeated_file(wlst_tmp_path)
del_repeated_file(wlst_path)

with open(wnids_path) as txt:
    for line in txt:
        label = line.strip('\n')
        with open(words_path) as dic:
            for dic_line in dic:
                idx_1 = dic_line.strip('\n').split('\t')[0]
                idx_2 = dic_line.strip('\n').split('\t')[1]
                if idx_1 == label:
                    with open(wlst_tmp_path, "a") as file:
                        file.write(str(idx_2)+"\n")
print("Generate i100_wordlist_tmp!")

# 处理 i100_wordlist_tmp.txt，删除每行逗号后的文字
with open(wlst_tmp_path) as txt:
    for line in txt:
        label_1 = line.strip('\n').split(',')[0]
        with open(wlst_path, "a") as file:
            file.write(str(label_1)+"\n")

# 替换' '为'_'
f = open(wlst_path,'r')
alllines = f.readlines()
f.close()
f = open(wlst_path,'w+')
for eachline in alllines:
    a=re.sub(' ', '_', eachline)
    f.writelines(a)
f.close()

print("Generate i100_wordlist!")

# 获取上位词直到顶端
# 构建词典
with open(words_path) as words:
    netdict = {}
    for line in words:
        netdict[line.strip('\n').split('\t')[1]] = line.strip('\n').split('\t')[0]

with open(words_path) as words:
    netdictmore = {}
    for line in words:
        for sub in line.strip('\n').split('\t')[1].split(", "):
            netdictmore[sub] = line.strip('\n').split('\t')[0]
print("dicts have been built")


# 获得全部label
all_labels = []
all_words = []
with open(wnids_path) as wnid:
    print("read wnids...")
    for line in wnid:
        all_labels.append(line.strip('\n'))
    print(len(all_labels))  #一共有100个分类
    
# 获得全部label对应的单词
with open(wlst_path) as wordlist:
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