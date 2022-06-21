# Multi-Granularity Regularized Re-Balancing for Class Incremental Learning

This repository provides a demo of MGRB.

### Environment

- Python 3.7
- Pytorch 1.8.1
- CUDA 11.2



### Requirement

See `requirements.txt` for details



### File organization

```
├── data                    # dataset and nodepathinfo
├── demo
    ├── save                # data preprocessing file
    ├── parser.py           # add parameters
    ├── loss.py             # return loss
    ├── cal_MG.py           # calculate the multi-granularity regularization term
    ├── util.py             # 工具函数
    └── ...
```