import os
import numpy as np
from nltk.corpus import wordnet as wn

miniimagenet_path = './processed'
wnids_path = './processed/wnids.txt'
words_path = "./processed/words.txt"
wlst_tmp_path = './processed/mini_wordlist_tmp.txt'
wlst_path = './processed/mini_wordlist.txt'
order_dict = {'n01532829': 0, 'n01558993': 1, 'n01704323': 2, 'n01749939': 3, 'n01770081': 4, 'n01843383': 5, 'n01855672': 6, 'n01910747': 7, 'n01930112': 8, 'n01981276': 9, 'n02074367': 10, 'n02089867': 11, 'n02091244': 12, 'n02091831': 13, 'n02099601': 14, 'n02101006': 15, 'n02105505': 16, 'n02108089': 17, 'n02108551': 18, 'n02108915': 19, 'n02110063': 20, 'n02110341': 21, 'n02111277': 22, 'n02113712': 23, 'n02114548': 24, 'n02116738': 25, 'n02120079': 26, 'n02129165': 27, 'n02138441': 28, 'n02165456': 29, 'n02174001': 30, 'n02219486': 31, 'n02443484': 32, 'n02457408': 33, 'n02606052': 34, 'n02687172': 35, 'n02747177': 36, 'n02795169': 37, 'n02823428': 38, 'n02871525': 39, 'n02950826': 40, 'n02966193': 41, 'n02971356': 42, 'n02981792': 43, 'n03017168': 44, 'n03047690': 45, 'n03062245': 46, 'n03075370': 47, 'n03127925': 48, 'n03146219': 49, 'n03207743': 50, 'n03220513': 51, 'n03272010': 52, 'n03337140': 53, 'n03347037': 54, 'n03400231': 55, 'n03417042': 56, 'n03476684': 57, 'n03527444': 58, 'n03535780': 59, 'n03544143': 60, 'n03584254': 61, 'n03676483': 62, 'n03770439': 63, 'n03773504': 64, 'n03775546': 65, 'n03838899': 66, 'n03854065': 67, 'n03888605': 68, 'n03908618': 69, 'n03924679': 70, 'n03980874': 71, 'n03998194': 72, 'n04067472': 73, 'n04146614': 74, 'n04149813': 75, 'n04243546': 76, 'n04251144': 77, 'n04258138': 78, 'n04275548': 79, 'n04296562': 80, 'n04389033': 81, 'n04418357': 82, 'n04435653': 83, 'n04443257': 84, 'n04509417': 85, 'n04515003': 86, 'n04522168': 87, 'n04596742': 88, 'n04604644': 89, 'n04612504': 90, 'n06794110': 91, 'n07584110': 92, 'n07613480': 93, 'n07697537': 94, 'n07747607': 95, 'n09246464': 96, 'n09256479': 97, 'n13054560': 98, 'n13133613': 99}

def del_repeated_file(file_path):
    if(os.path.exists(file_path) == True):
        os.remove(file_path)

# 生成wnids.txt
i = 0
del_repeated_file(wnids_path)
del_repeated_file(wlst_tmp_path)

for key in order_dict:
    i = i + 1
    value = order_dict[key]
    with open(wnids_path, 'a') as f:
        f.write(key + "\n")
print("{} classes in total".format(i))
print("generate wnids.txt")

# 生成wordlist.txt
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

print("Generate mini_wordlist_tmp.txt!")

# 处理mini_wordlist_tmp.txt，每行删除逗号后的文字
with open(wlst_tmp_path) as txt:
    for line in txt:
        label_1 = line.strip('\n').split(',')[0]
        with open(wlst_path, "a") as file:
            file.write(str(label_1)+"\n")

# 替换' '为'_'
import re
f = open(wlst_path,'r')
alllines = f.readlines()
f.close()
f = open(wlst_path,'w+')
for eachline in alllines:
    a=re.sub(' ', '_', eachline)
    f.writelines(a)
f.close()

print("Finished!")

# 构建词典
with open(words_path) as words:
    netdict = {}
    for line in words:
        netdict[line.strip('\n').split('\t')[1]] = line.strip('\n').split('\t')[0]
'''
print(netdict)
{'entity': 'n00001740', 'physical entity': 'n00001930', 'abstraction, abstract entity': 'n00002137', 'thing': 'n13943968',...}
'''
with open(words_path) as words:
    netdictmore = {}
    for line in words:
        for sub in line.strip('\n').split('\t')[1].split(", "):
            netdictmore[sub] = line.strip('\n').split('\t')[0]
print("dicts have been built")
'''
print(netdictmore["abstract entity"])
{'abstraction':n05854150, 'abstract entity': n00002137, ...}
'''


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
    print("read re-mini_wordlist...")
    for line in wordlist:
        all_words.append(line.strip('\n'))
    print(len(all_words))  #一共有100个分类

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
    # 含义不对应的词，特殊标记并在下方进行更正
    if i == 41:
        cat = "carousel.n.02"
    elif i == 80:
        cat = "stage.n.03"
    elif i == 99:
        cat = "ear.n.05"
    elif i == 67:
        cat = "organ.n.05"
    elif i == 94:
        cat = "hotdog.n.02"
    cat = wn.synset(cat).hypernyms()  #　【cat变量】的上位词
    # print(cat)
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
np.save('./processed/nodepathinfo.npy', nodepath)
print("Finish!")