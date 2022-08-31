# Multi-Granularity Regularized Re-Balancing for Class Incremental Learning

### Introduction

Data imbalance between old and new classes is a key issue that leads to performance degradation of the model in incremental learning. In this study, we propose an assumption-agnostic method, Multi-Granularity Regularized re-Balancing (MGRB), to address this problem. Re-balancing methods are used to alleviate the influence of data imbalance; however, we empirically discover that they would under-fit new classes. To this end, we further design a novel multi-granularity regularization term that enables the model to consider the correlations of classes in addition to re-balancing the data. A class hierarchy is first constructed by ontology, grouping semantically or visually similar classes. The multi-granularity regularization then transforms the one-hot label vector into a continuous label distribution, which reflects the relations between the target class and other classes based on the constructed class hierarchy. Thus, the model can learn the inter-class relational information, which helps enhance the learning of both old and new classes. Experimental results on both public datasets and a real-world fault diagnosis dataset verify the effectiveness of the proposed method. 



![model_v9](E:\研究生\论文\MGRB\arxiv版本\model_v9.png)



> The figure on the left shows the structure of the baseline method, and the figure on the right is the overview of our MGRB method. The proposed model has two significant parts compared with the baseline: (1) re-balancing modeling. We use the re-balancing strategies during training to alleviate the influence of data imbalance; and (2) multi-granularity regularization. A multi-granularity regularization term is designed to make the model consider class correlations. Through end-to-end learning, both old and new classes can be better learned.

### Environment

- Python 3.7
- Pytorch 1.8.1
- CUDA 11.2

### Requirement

+ See `requirements.txt` for environment.

+ The pre-trained word vector library can be found [here](https://code.google.com/archive/p/word2vec/). You can download it or download the library used in this paper [here](https://drive.google.com/file/d/1xZEbpkDXZF_rlH9hBIq-WE_RQ5sNg3hp/view?usp=sharing).

### File organization

```
├── data                    # dataset and nodepathinfo
├── demo
    ├── save                # data preprocessing file
    ├── parser.py           # add parameters
    ├── loss.py             # return loss
    ├── cal_MG.py           # calculate the multi-granularity regularization term
    ├── util.py             
    └── ...
├── code
```



If you use this paper/code in your research, please consider citing us: