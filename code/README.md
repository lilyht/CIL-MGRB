# Multi-Granularity Regularized Re-Balancing for Class Incremental Learning

 

#### Step1: Split the dataset and generate the file

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



#### Step2: Running Experiments

To run experiments on CIFAR10 with hierarchy clustering or semantic hierarchy clustering:

~~~bash
python -u etrain.py -ds cifar10 -net resnet -n 5 -m 1 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss -cluster 2 -beta 20 -dscale 1.0 -scale 0.5  -savename c10_sem_5_1 > c10_sem_5_1.txt 2>&1
python -u etrain.py -ds cifar10 -net resnet -n 5 -m 1 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss --vis_hier -cluster 5 -beta 20 -dscale 1.0 -scale 0.5 -savename c10_vis_5_1 > c10_vis_5_1.txt 2>&1
~~~

To run experiments on CIFAR100 with ontology and visual hierarchical clustering:

~~~bash
python dealcifar100.py -n 10 -m 10
python -u etrain.py -ds cifar100 -net resnet -n 10 -m 10 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss -beta 20 -dscale 0.1 -savename c100_ont_10_10 > c100_ont_10_10.txt 2>&1
python -u etrain.py -ds cifar100 -net resnet -n 10 -m 10 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss --vis_hier -cluster 5 -beta 20 -dscale 0.1 -savename c100_vis_10_10_n5 > c100_vis_10_10_n5.txt 2>&1
~~~

To run experiments on miniImageNet with ontology, hierarchy clustering or semantic hierarchy clustering:

~~~bash
python -u etrain.py -ds miniimagenet -net resnet -n 50 -m 10 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss -beta 50 -cluster -2 -dscale 1.0 -savename mini_ont_50_10 > mini_ont_50_10.txt 2>&1
python -u etrain.py -ds miniimagenet -net resnet -n 50 -m 10 -bs 128 -gpu 0 -lr 0.001 -lrdecay 0.1 -setloss ce+distill+hierloss --cbloss --vis_hier -cluster 10 -beta 50 -dscale 1.0 -savename mini_vis_50_10_n10 > mini_vis_50_10_n10.txt 2>&1
~~~



Furthermore several commands are available to reproduce the experiments showcased in the paper. Please see the file `./code/run.sh`.



If you use this paper/code in your research, please consider citing us: