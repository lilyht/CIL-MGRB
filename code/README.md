# Multi-Granularity Regularized Re-Balancing for Class Incremental Learning

 

#### Step1: split the dataset and generate the file

Please run the following command to get the dataset partition files, where the parameter `-k` represents the number of samples can be stored in the final phase (the default value is 2000), `-n` represents the number of classes that can be learned in the initial stage, and `-m` represents the number of new classes appear in every new round.

This process will generate  `xxx_divide_info_n_m.npy`  and  `xxx_num_info_n_m.npy`  for further use.

For CIFAR , running the following command will generate data files in the folder `./data/cifarxxx/`

+ For CIFAR10

  ~~~bash
  python dealcifar10.py -k 2000 -n 50 -m 10
  ~~~

+ For CIFAR100

  ~~~bash
  python dealcifar100.py -k 2000 -n 50 -m 10
  ~~~

+ For miniImageNet

  ~~~bash
  python divideminiimagenet.py -k 2000 -n 50 -m 10
  ~~~

+ For ImageNetSubset

  ~~~bash
  python divideimagenet100.py -k 2000 -n 50 -m 10
  ~~~



#### Step2: train

