# Multi-Granularity Regularized Re-Balancing for Class Incremental Learning

This repository provides a demo of MGRB.  The component of multi-granularity regularization term can be found in `cal_MG.py`. 

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
```