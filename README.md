# Awesome Meta-Learning Papers ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A summary of meta learning papers based on realm. Sorted by submission date on arXiv.

# [Topics]()

* [Survey](##Survey)
* [Few-shot learning](##Few-shot learning)
* [Reinforcement Learning](## Reinforcement learning)
* [AutoML](## AutoML)
* [Task-dependent Methods](## Task-dependent)
* [Data Aug & Reg](## Data Aug & Reg)
* [Lifelong learning](## Lifelong learning)
* [Domain generalization](## Domain generalization)     
* [Neural process](## Neural process)
* [Configuration transfer (Adaptation， Hyperparameter Opt)](## Configuration transfer (Adaptation， Hyperparameter Opt))
* [Model compression](## Model compression)
* [Kernel learning](## Kernel learning)
* [Robustness](## Robustness)
* [Bayesian inference](## Bayesian inference)
* [Optimization](## Optimization)
* [Theory](## Theory)

## [Survey]()
Meta-Learning in Neural Networks: A Survey [[paper](https://arxiv.org/abs/2004.05439)]
  - Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey 

Meta-Learning[[paper](https://www.ml4aad.org/wp-content/uploads/2018/09/chapter2-metalearning.pdf)]
  - Joaquin Vanschoren

Meta-Learning: A Survey [[paper](https://arxiv.org/pdf/1810.03548.pdf)]
  - Joaquin Vanschoren

Meta-learners’ learning dynamics are unlike learners’  [[paper](https://arxiv.org/pdf/1905.01320.pdf)]
  - Neil C. Rabinowitz

## [Few-shot learning]()

Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification [[paper](https://arxiv.org/abs/2204.04567)]
  - Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang, Peihua Li --CVPR 2022

Learning Prototype-oriented Set Representations for Meta-Learning [[paper](https://openreview.net/forum?id=WH6u2SvlLp4)]
  - Dan dan Guo, Long Tian, Minghe Zhang, Mingyuan Zhou, Hongyuan Zha --ICLR 2022

BOIL: Towards Representation Change for Few-shot Learning [[paper](https://openreview.net/forum?id=umIdUL8rMH)]
  - Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, Se-Young Yun --ICLR 2021

Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels [[paper](https://arxiv.org/abs/1910.05199)]
  - Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O'Boyle, Amos Storkey --NeurIPS 2020

Laplacian Regularized Few-Shot Learning [[paper](http://proceedings.mlr.press/v119/ziko20a/ziko20a.pdf)]
  - Imtiaz Masud Ziko, Jose Dolz, Eric Granger, Ismail Ben Ayed --ICML 2020

Few-shot Sequence Learning with Transformer 
  -  Lajanugen Logeswaran, Ann Lee, Myle Ott, Honglak Lee, Marc´Aurelio Ranzato, Arthur Szlam --NeurIPS 2020 #Meta-Learning

Prototype Rectification for Few-Shot Learning [[paper](https://arxiv.org/abs/1911.10713)]
  - Jinlu Liu, Liang Song, Yongqiang Qin --ECCV 2020

When Does Self-supervision Improve Few-shot Learning? [[paper](https://arxiv.org/pdf/1910.03560.pdf)]
  - Jong-Chyi Su, Subhransu Maji, Bharath Hariharan --ECCV 2020
  
Cross Attention Network for Few-shot Classification [[paper](https://arxiv.org/abs/1910.07677)]
  - Ruibing Hou, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen --NeurIPS 2019

Learning to Learn via Self-Critique [[paper](https://arxiv.org/abs/1905.10295)]
  - Antreas Antoniou, Amos Storkey  --NeurIPS 2019
  
Learning from the Past: Continual Meta-Learning with Bayesian Graph Neural Networks [[paper](https://arxiv.org/abs/1911.04695)]
  - Yadan Luo, Zi Huang, Zheng Zhang, Ziwei Wang, Mahsa Baktashmotlagh, Yang Yang --AAAI 2020

Few-Shot Learning with Global Class Representations [[paper](https://arxiv.org/pdf/1908.05257.pdf)]
  - Tiange Luo, Aoxue Li, Tao Xiang, Weiran Huang, Liwei Wang  --ICCV 2019

TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning [[paper](https://arxiv.org/pdf/1905.06549.pdf)]
  - Sung Whan Yoon, Jun Seo, Jaekyun Moon --ICML 2019

Learning to Learn with Conditional Class Dependencies  [[paper](https://openreview.net/pdf?id=BJfOXnActQ)]
  - Xiang Jiang, Mohammad Havaei, Farshid Varno, Gabriel Chartrand, Nicolas Chapados, Stan Matwin --ICLR 2019
 
 Finding Task-Relevant Features for Few-Shot Learning by Category Traversal [[paper](https://arxiv.org/pdf/1905.11116.pdf)]
  - Hongyang Li, David Eigen, Samuel Dodge, Matthew Zeiler, Xiaogang Wang --CVPR 2019
 
TAFE-Net: Task-Aware Feature Embeddings for Low Shot Learning [[paper](https://arxiv.org/abs/1904.05967)]
  - Xin Wang, Fisher Yu, Ruth Wang, Trevor Darrell, Joseph E. Gonzalez --CVPR 2019

Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images [[paper](https://arxiv.org/abs/1904.08482)]
  - Junsik Kim, Tae-Hyun Oh, Seokju Lee, Fei Pan, In So Kweon --CVPR 2019

LCC: Learning to Customize and Combine Neural Networks for Few-Shot Learning [[paper](https://arxiv.org/pdf/1904.08479.pdf)]
  - Yaoyao Liu, Qianru Sun, An-An Liu, Yuting Su, Bernt Schiele, Tat-Seng Chua --CVPR 2019

Meta-Learning with Differentiable Convex Optimization [[paper](https://arxiv.org/abs/1904.03758)]
  - Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, Stefano Soatto --CVPR 2019

Dense Classification and Implanting for Few-Shot Learning [[paper](https://arxiv.org/pdf/1903.05050.pdf)]
  - Yann Lifchitz, Yannis Avrithis, Sylvaine Picard, Andrei Bursuc --CVPR 2019

Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples
  - Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo Larochelle  -- arXiv 2019
 
Adaptive Cross-Modal Few-Shot Learning [[paper](https://arxiv.org/pdf/1902.07104.pdf)]
  - Chen Xing, Negar Rostamzadeh, Boris N. Oreshkin, Pedro O. Pinheiro --arXiv 2019

Meta-Learning with Latent Embedding Optimization [[paper](https://arxiv.org/abs/1807.05960)]
  - Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, Raia Hadsell -- ICLR 2019
 
A Closer Look at Few-shot Classification [[paper](https://arxiv.org/abs/1904.04232)]
  - Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang  -- ICLR 2019
 
Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning [[paper](https://arxiv.org/pdf/1805.10002.pdf)]
  - Yanbin Liu, Juho Lee, Minseop Park, Saehoon Kim, Eunho Yang, Sung Ju Hwang, Yi Yang -- ICLR 2019

Dynamic Few-Shot Visual Learning without Forgetting [[paper](https://arxiv.org/pdf/1804.09458v1.pdf)]
  - Spyros Gidaris, Nikos Komodakis --arXiv 2019

Meta Learning with Lantent Embedding Optimization [[paper](https://openreview.net/pdf?id=BJgklhAcK7)]
  - Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero & Raia Hadsell --ICLR 2019

Adaptive Posterior Learning: few-shot learning with a surprise-based memory module
  - Tiago Ramalho, Marta Garnelo --ICLR 2019

How To Train Your MAML [[paper](https://arxiv.org/pdf/1810.09502v1.pdf)]
  - Antreas Antoniou, Harrison Edwards, Amos Storkey -- ICLR 2019

TADAM: Task dependent adaptive metric for improved few-shot learning [[paper](https://arxiv.org/abs/1805.10123)]
  - Boris N. Oreshkin, Pau Rodriguez, Alexandre Lacoste --arXiv 2019

Few-shot Learning with Meta Metric Learners
  - Yu Cheng, Mo Yu, Xiaoxiao Guo, Bowen Zhou --NIPS 2017 workshop on Meta-Learning

Learning Embedding Adaptation for Few-Shot Learning [[paper](https://arxiv.org/pdf/1812.03664.pdf)]
  - Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, Fei Sha --arXiv 2018

Meta-Transfer Learning for Few-Shot Learning [[paper](https://arxiv.org/pdf/1812.02391.pdf)]
  - Qianru Sun, Yaoyao Liu, Tat-Seng Chu, Bernt Schiele -- arXiv 2018

Task-Agnostic Meta-Learning for Few-shot Learning
  - Muhammad Abdullah Jamal, Guo-Jun Qi, and Mubarak Shah  --arXiv 2018

Few-Shot Learning with Graph Neural Networks [[paper](https://arxiv.org/abs/1711.04043)]
  - Victor Garcia, Joan Bruna -- ICLR 2018

Prototypical Networks for Few-shot Learning [[paper](https://arxiv.org/pdf/1703.05175.pdf)]
  - Jake Snell, Kevin Swersky, Richard S. Zemel -- NIPS 2017
  
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [[paper](https://arxiv.org/abs/1703.03400)]
  - Chelsea Finn, Pieter Abbeel, Sergey Levine -- ICML 2016

### Large scale dataset
Image Deformation Meta-Networks for One-Shot Learning [[paper](https://arxiv.org/pdf/1905.11641.pdf)]
  - Zitian Chen, Yanwei Fu, Yu-Xiong Wang, Lin Ma, Wei Liu, Martial Hebert --CVPR 2019

### Imbalance class
Balanced Meta-Softmax for Long-Tailed Visual Recognition [[paper](https://arxiv.org/abs/2007.10740)]
  - Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li --NeurIPS 2020

MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler [[paper](https://proceedings.neurips.cc/paper/2020/file/a64bd53139f71961c5c31a9af03d775e-Paper.pdf)]
  - Zhining Liu, Pengfei Wei, Jing Jiang, Wei Cao, Jiang Bian, Yi Chang --NeurIPS 2019

Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks [[paper](https://openreview.net/pdf?id=rkeZIJBYvr)]
  - Donghyun Na, Hae Beom Lee, Hayeon Lee, Saehoon Kim, Minseop Park, Eunho Yang, Sung Ju Hwang --ICLR 2020

Meta-weight-net: Learning an explicit mapping for sample weighting [[paper](https://papers.nips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)]
  - Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, Deyu Meng --NeurIPS 2019

Learning to Reweight Examples for Robust Deep Learning [[paper](https://arxiv.org/pdf/1803.09050.pdf)]
  - Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun --ICML 2018

Learning to Model the Tail [[paper](https://papers.nips.cc/paper/7278-learning-to-model-the-tail.pdf)]
  - Yu-Xiong Wang, Deva Ramanan, Martial Hebert --NeurIPS 2017

### Video retargeting
MetaPix: Few-Shot Video Retargeting [[paper](https://openreview.net/forum?id=SJx1URNKwH)]
  - Jessica Lee, Deva Ramanan, Rohit Girdhar --ICLR 2020

### Object detection
Few-shot Object Detection via Feature Reweighting [[paper](https://arxiv.org/abs/1812.01866)]
  - Bingyi Kang, Zhuang Liu, Xin Wang, Fisher Yu, Jiashi Feng, Trevor Darrell --ICCV 2019

### Segmentation
PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment [[paper](https://arxiv.org/abs/1908.06391)]
  - Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, Jiashi Feng --ICCV 2019

### NLP
Meta-Learning for Few-Shot NMT Adaptation [[paper](https://arxiv.org/abs/2004.02745)]
  - Amr Sharaf, Hany Hassan, Hal Daumé III --arXiv 2020

Learning to Few-Shot Learn Across Diverse Natural Language Classification Tasks [[paper](https://arxiv.org/pdf/1911.03863.pdf)]
  - Trapit Bansal, Rishikesh Jha, Andrew McCallum  --arXiv 2020

Compositional generalization through meta sequence-to-sequence learning [[paper](https://arxiv.org/abs/1906.05381)]
  - Brenden M. Lake  --NeurIPS 2019

Few-Shot Representation Learning for Out-Of-Vocabulary Words [[paper](https://arxiv.org/abs/1907.00505)]
  - Ziniu Hu, Ting Chen, Kai-Wei Chang, Yizhou Sun --ACL 2019


## Reinforcement learning

Offline Meta-Reinforcement Learning with Online Self-Supervision [[paper](https://arxiv.org/abs/2107.03974)]
  - Vitchyr Pong, Ashvin Nair, Laura Smith, Catherine Huang, Sergey Levine  --ICML 2022

System-Agnostic Meta-Learning for MDP-based Dynamic Scheduling via Descriptive Policy [[paper](https://arxiv.org/abs/2201.07051)]
  - Lee, Hyun-Suk --AISTATS 2022

Meta Learning MDPs with Linear Transition Models [[paper](https://arxiv.org/abs/2201.08732)]
  - Müller, Robert ; Pacchiano, Aldo --AISTATS 2022

CoMPS: Continual Meta Policy Search [[paper](https://openreview.net/forum?id=PVJ6j87gOHz)]
  - Glen Berseth, Zhiwei Zhang, Grace Zhang, Chelsea Finn, Sergey Levine --ICLR 2022

Modeling and Optimization Trade-off in Meta-learning [[paper](https://proceedings.neurips.cc/paper/2020/hash/7fc63ff01769c4fa7d9279e97e307829-Abstract.html)]
  - Katelyn Gao, Ozan Sener --NeurIPS 2020

Information-theoretic Task Selection for Meta-Reinforcement Learning [[paper](https://arxiv.org/abs/2011.01054)]
  - Ricardo Luna Gutierrez, Matteo Leonetti --NeurIPS 2020

On the Global Optimality of Model-Agnostic Meta-Learning: Reinforcement Learning and Supervised Learning [[paper](https://proceedings.icml.cc/static/paper_files/icml/2020/1816-Paper.pdf)]
  - Lingxiao Wang, Qi Cai, Zhuoyan Yang, Zhaoran Wang --PMLR 2020

Generalized Reinforcement Meta Learning for Few-Shot Optimization [[paper](https://arxiv.org/abs/2005.01246)]
  - Raviteja Anantha, Stephen Pulman, Srinivas Chappidi --ICML 2020

VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning [[paper](https://openreview.net/forum?id=Hkl9JlBYvr)]
  - Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, Shimon Whiteson --ICLR 2020

Reinforcement Learning with Competitive Ensembles of Information-Constrained Primitives [[paper](https://openreview.net/forum?id=ryxgJTEYDr)]
  - Anirudh Goyal, Shagun Sodhani, Jonathan Binas, Xue Bin Peng, Sergey Levine, Yoshua Bengio --ICLR 2020

Meta-learning curiosity algorithms [[paper](https://openreview.net/pdf?id=BygdyxHFDS)]
  - Ferran Alet*, Martin F. Schneider*, Tomas Lozano-Perez, Leslie Pack Kaelbling --ICLR 2020

Meta-Q-Learning [[paper](https://openreview.net/forum?id=SJeD3CEFPH)]
  - Rasool Fakoor, Pratik Chaudhari, Stefano Soatto, Alexander J. Smola --ICLR 2020
  
Guided Meta-Policy Search [[paper](https://arxiv.org/abs/1904.00956)]
  - Russell Mendonca, Abhishek Gupta, Rosen Kralev, Pieter Abbeel, Sergey Levine, Chelsea Finn

## AutoML

Learning meta-features for AutoML [[paper](https://openreview.net/forum?id=DTkEfj0Ygb8)]
  - Herilalaina Rakotoarison, Louisot Milijaona, Andry RASOANAIVO, Michele Sebag, Marc Schoenauer --ICLR 2022

Towards Fast Adaptation of Neural Architectures with Meta Learning [[paper](https://openreview.net/forum?id=r1eowANFvr)]
  - Dongze Lian, Yin Zheng, Yintao Xu, Yanxiong Lu, Leyu Lin, Peilin Zhao, Junzhou Huang, Shenghua Gao --ICLR 2020

Graph HyperNetworks for Neural Architecture Search [[paper](https://arxiv.org/abs/1810.05749)]
  - Chris Zhang, Mengye Ren, Raquel Urtasun --ICLR 2019

Fast Task-Aware Architecture Inference
  - Efi Kokiopoulou, Anja Hauth, Luciano Sbaiz, Andrea Gesmundo, Gabor Bartok, Jesse Berent --arXiv 2019

Bayesian Meta-network Architecture Learning
  - Albert Shaw, Bo Dai, Weiyang Liu, Le Song --arXiv 2018
  
## Task-dependent

Meta-Learning with Fewer Tasks through Task Interpolation [[paper](https://openreview.net/forum?id=ajXWF7bVR8d)]
  - Huaxiu Yao, Linjun Zhang, Chelsea Finn --ICLR 2022

Meta-Regularization by Enforcing Mutual-Exclusiveness [[paper](https://arxiv.org/abs/2101.09819)]
  - Edwin Pan, Pankaj Rajak, Shubham Shrivastava --arXiv 2021

Task-Robust Model-Agnostic Meta-Learning [[paper](https://papers.nips.cc/paper/2020/file/da8ce53cf0240070ce6c69c48cd588ee-Paper.pdf)]
  - Liam Collins, Aryan Mokhtari, Sanjay Shakkottai --NeurIPS 2020

Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation [[paper](https://arxiv.org/abs/1910.13616)]
  - Risto Vuorio, Shao-Hua Sun, Hexiang Hu, Joseph J. Lim --NeurIPS 2019

Meta-Learning with Warped Gradient Descent [[paper](https://arxiv.org/pdf/1909.00025.pdf)]
  - Sebastian Flennerhag, Andrei A. Rusu, Razvan Pascanu, Hujun Yin, Raia Hadsell --arXiv 2019

TAFE-Net: Task-Aware Feature Embeddings for Low Shot Learning [[paper](https://arxiv.org/abs/1904.05967)]
  - Xin Wang, Fisher Yu, Ruth Wang, Trevor Darrell, Joseph E. Gonzalez --CVPR 2019

TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning [[paper](https://arxiv.org/pdf/1905.06549.pdf)]
  - Sung Whan Yoon, Jun Seo, Jaekyun Moon --ICML 2019

Meta-Learning with Latent Embedding Optimization [[paper](https://arxiv.org/abs/1807.05960)]
  - Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, Raia Hadsell -- ICLR 2019

Fast Task-Aware Architecture Inference
  - Efi Kokiopoulou, Anja Hauth, Luciano Sbaiz, Andrea Gesmundo, Gabor Bartok, Jesse Berent --arXiv 2019

Task2Vec: Task Embedding for Meta-Learning
  - Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless Fowlkes, Stefano Soatto, Pietro Perona--arXiv 2019

TADAM: Task dependent adaptive metric for improved few-shot learning
  - Boris N. Oreshkin, Pau Rodriguez, Alexandre Lacoste --arXiv 2019
  
MetaReg: Towards Domain Generalization using Meta-Regularization [[paper](https://papers.nips.cc/paper/7378-metareg-towards-domain-generalization-using-meta-regularization.pdf)]
  - Yogesh Balaji, Swami Sankaranarayanan -- NIPS 2018

### Heterogeneous task
Statistical Model Aggregation via Parameter Matching [[paper]](https://arxiv.org/pdf/1911.00218.pdf)
  - Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Trong Nghia Hoang --NeurIPS 2019

Hierarchically Structured Meta-learning [[paper](https://arxiv.org/pdf/1905.05301.pdf)]
  - Huaxiu Yao, Ying Wei, Junzhou Huang, Zhenhui Li --ICML 2019

Hierarchical Meta Learning [[paper](https://arxiv.org/abs/1904.09081)]
  - Yingtian Zou, Jiashi Feng  --arXiv 2019


## Data Aug & Reg
MetAug: Contrastive Learning via Meta Feature Augmentation [[paper](https://arxiv.org/abs/2203.05119)]
  - Jiangmeng Li, Wenwen Qiang, Changwen Zheng, Bing Su, Hui Xiong --ICML 2022

MetaInfoNet: Learning Task-Guided Information for Sample Reweighting [[paper](https://arxiv.org/abs/2012.05273)]
  - Hongxin Wei, Lei Feng, Rundong Wang, Bo An --arXiv 2020

Meta Dropout: Learning to Perturb Latent Features for Generalization [[paper](https://openreview.net/forum?id=BJgd81SYwr)]
  - Hae Beom Lee, Taewook Nam, Eunho Yang, Sung Ju Hwang --ICLR 2020

Learning to Reweight Examples for Robust Deep Learning [[paper](https://arxiv.org/abs/1803.09050)]
  - Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun --ICML 2018


## Lifelong learning

Optimizing Reusable Knowledge for Continual Learning via Metalearning [[paper](https://arxiv.org/abs/2106.05390)]
  - Julio Hurtado, Alain Raymond-Saez, Alvaro Soto --NeurIPS 2021

Learning where to learn: Gradient sparsity in meta and continual learning [[paper](https://arxiv.org/pdf/2110.14402.pdf)]
  - Johannes von Oswald, Dominic Zhao, Seijin Kobayashi, Simon Schug, Massimo Caccia, Nicolas Zucchet, João Sacramento --NeurIPS 2021

Online-Within-Online Meta-Learning [[paper](https://papers.nips.cc/paper/9468-online-within-online-meta-learning)]
  - Giulia Denevi, Dimitris Stamos, Carlo Ciliberto, Massimiliano Pontil
  
Reconciling meta-learning and continual learning with online mixtures of tasks [[paper](https://arxiv.org/abs/1812.06080)]
  - Ghassen Jerfel, Erin Grant, Thomas L. Griffiths, Katherine Heller  --NeurIPS 2019

Meta-Learning Representations for Continual Learning [[paper](https://arxiv.org/abs/1905.12588)]
  - Khurram Javed, Martha White  --NeurIPS 2019

Online Meta-Learning [[paper](https://arxiv.org/abs/1902.08438)]
  - Chelsea Finn, Aravind Rajeswaran, Sham Kakade, Sergey Levine  --ICML 2019

Hierarchically Structured Meta-learning [[paper](https://arxiv.org/pdf/1905.05301.pdf)]
  - Huaxiu Yao, Ying Wei, Junzhou Huang, Zhenhui Li --ICML 2019

A Neural-Symbolic Architecture for Inverse Graphics Improved by Lifelong Meta-Learning [[paper](https://arxiv.org/pdf/1905.08910.pdf)]
  - Michael Kissner, Helmut Mayer --arXiv 2019

Incremental Learning-to-Learn with Statistical Guarantees [[paper](http://auai.org/uai2018/proceedings/papers/181.pdf)]
  - Giulia Denevi, Carlo Ciliberto, Dimitris Stamos, Massimiliano Pontil --arXiv 2018
  
## Domain generalization
Meta-learning curiosity algorithms [[paper](https://openreview.net/pdf?id=BygdyxHFDS)]
  - Ferran Alet*, Martin F. Schneider*, Tomas Lozano-Perez, Leslie Pack Kaelbling --ICLR 2020

Domain Generalization via Model-Agnostic Learning of Semantic Features [[paper](https://arxiv.org/abs/1910.13580)]
  - Qi Dou, Daniel C. Castro, Konstantinos Kamnitsas, Ben Glocker
  
Learning to Generalize: Meta-Learning for Domain Generalization [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16067/16547)]
  - Da Li, Yongxin Yang, Yi-Zhe Song, Timothy M. Hospedales --AAAI 2018

## Bayesian inference

Stochastic Deep Networks with Linear Competing Units for Model-Agnostic Meta-Learning [[paper](https://openreview.net/forum?id=FFGDKzLasUa)]
  - Konstantinos Ι. Kalais, Sotirios Chatzis --ICML 2022

Meta-Learning with Variational Bayes [[paper](https://arxiv.org/abs/2103.02265)]
  - Lucas D. Lingle --arXiv 2021

Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization [[paper](https://openreview.net/forum?id=ryeYpJSKwr)]
  - Michael Volpp, Lukas Froehlich, Kirsten Fischer, Andreas Doerr, Stefan Falkner, Frank Hutter, Christian Daniel --ICLR 2020

Bayesian Meta Sampling for Fast Uncertainty Adaptation [[paper](https://openreview.net/forum?id=Bkxv90EKPB)]
  - Zhenyi Wang, Yang Zhao, Ping Yu, Ruiyi Zhang, Changyou Chen --ICLR 2020

Meta-Learning Mean Functions for Gaussian Processes [[paper](https://arxiv.org/pdf/1901.08098.pdf)]
  - Vincent Fortuin, Heiko Strathmann, and Gunnar Rätsch --NeurIPS 2019 workshop

Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks [[paper](https://openreview.net/pdf?id=rkeZIJBYvr)]
  - Donghyun Na, Hae Beom Lee, Hayeon Lee, Saehoon Kim, Minseop Park, Eunho Yang, Sung Ju Hwang --ICLR 2020

Meta-Learning without Memorization [[paper](https://openreview.net/pdf?id=BklEFpEYwS)]
  - Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, Chelsea Finn --ICLR 2020

Meta-Amortized Variational Inference and Learning [[paper](https://arxiv.org/pdf/1902.01950.pdf)]
  - Mike Wu, Kristy Choi, Noah Goodman, Stefano Ermon  --arXiv 2019

Amortized Bayesian Meta-Learning [[paper](https://openreview.net/pdf?id=rkgpy3C5tX)]
  - Sachin Ravi, Alex Beatson --ICLR 2019
  
Neural Processes [[paper](https://arxiv.org/abs/1807.01622)]
  - Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh 

Meta-Learning Probabilistic Inference For Prediction [[paper](https://arxiv.org/pdf/1805.09921.pdf)]
  - Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, Richard E. Turner --ICLR 2019

Meta-Learning Priors for Efficient Online Bayesian Regression [[paper](https://arxiv.org/abs/1807.08912)]
  - James Harrison, Apoorva Sharma, Marco Pavone --WAFR 2018

Probabilistic Model-Agnostic Meta-Learning [[paper](https://arxiv.org/pdf/1806.02817.pdf)]
  - Chelsea Finn, Kelvin Xu, Sergey Levine  --arXiv 2018

Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions [[paper](https://arxiv.org/abs/1710.10304)]
  - Scott Reed, Yutian Chen, Thomas Paine, Aäron van den Oord, S. M. Ali Eslami, Danilo Rezende, Oriol Vinyals, Nando de Freitas --ICLR 2018

Bayesian Model-Agnostic Meta-Learning [[paper](https://arxiv.org/abs/1806.03836)]
  - Taesup Kim, Jaesik Yoon, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, Sungjin Ahn -- NIPS 2018

Meta-learning by adjusting priors based on extended PAC-Bayes theory [[paper](https://arxiv.org/pdf/1711.01244.pdf)]
  - Ron Amit , Ron Meir --ICML 2018

## Neural process

Neural Variational Dropout Processes [[paper](https://openreview.net/forum?id=lyLVzukXi08)]
  - Insu Jeon, Youngjin Park, Gunhee Kim --ICLR 2022

Neural ODE Processes [[paper](https://arxiv.org/abs/2103.12413)]
  - Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, Pietro Liò --ICLR 2021

Convolutional Conditional Neural Processes [[paper](https://openreview.net/forum?id=Skey4eBYPS)]
  - Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, Richard E. Turner --ICLR 2020

Bootstrapping Neural Processes [[paper](https://arxiv.org/pdf/2008.02956.pdf)]
  - Juho Lee, Yoonho Lee, Jungtaek Kim, Eunho Yang, Sung Ju Hwang, Yee Whye Teh --NeurIPS 2020

MetaFun: Meta-Learning with Iterative Functional Updates [[paper](http://proceedings.mlr.press/v119/xu20i/xu20i.pdf)]
  - Jin Xu, Jean-Francois Ton, Hyunjik Kim, Adam R. Kosiorek, Yee Whye Teh --ICML 2020

Sequential Neural Processes [[paper](https://arxiv.org/abs/1906.10264)]
  - Gautam Singh, Jaesik Yoon, Youngsung Son, Sungjin Ahn --NeurIPS 2019 

Neural Processes [[paper](https://arxiv.org/abs/1807.01622)]
  - Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh --arXiv 2018

Conditional Neural Processes [[paper](https://arxiv.org/abs/1807.01613)]
  - Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami --ICML 2018

## Configuration transfer (Adaptation， Hyperparameter Opt)

Online Hyperparameter Meta-Learning with Hypergradient Distillation [[paper](https://openreview.net/forum?id=01AMRlen9wJ)]
  - Hae Beom Lee, Hayeon Lee, JaeWoong Shin, Eunho Yang, Timothy Hospedales, Sung Ju Hwang --ICLR 2022

Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels [[paper](https://arxiv.org/abs/1910.05199)]
  - Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O'Boyle, Amos Storkey --NeurIPS 2020

Meta-Learning for Few-Shot NMT Adaptation [[paper](https://www.aclweb.org/anthology/2020.ngt-1.5.pdf)]
  - Amr Sharaf, Hany Hassan, Hal Daumé III --arXiv 2020

Fast Context Adaptation via Meta-Learning [[paper](https://arxiv.org/pdf/1810.03642.pdf)]
  - Luisa M Zintgraf, Kyriacos Shiarlis, Vitaly Kurin, Katja Hofmann, Shimon Whiteson  --ICML 2019

Zero-Shot Knowledge Distillation in Deep Networks [[paper](https://arxiv.org/pdf/1905.08114.pdf)]
  - Gaurav Kumar Nayak *, Konda Reddy Mopuri, Vaisakh Shaj, R. Venkatesh Babu, Anirban Chakraborty --ICML 2019

Toward Multimodal Model-Agnostic Meta-Learning [[paper](https://arxiv.org/pdf/1812.07172.pdf)]
  - Risto Vuorio, Shao-Hua Sun, Hexiang Hu, Joseph J. Lim  --arXiv 2019

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [[paper](https://arxiv.org/abs/1703.03400)]
  - Chelsea Finn, Pieter Abbeel, Sergey Levine -- ICML 2016

### Semi/Unsupervised learning

Unsupervised Learning via Meta-Learning [[paper](https://arxiv.org/abs/1810.02334)]
  - Kyle Hsu, Sergey Levine, Chelsea Finn -- ICLR 2019
  
Meta-Learning Update Rules for Unsupervised Representation Learning [[paper](https://openreview.net/pdf?id=HkNDsiC9KQ)]
  - Luke Metz, Niru Maheswaranathan, Brian Cheung, Jascha Sohl-Dickstein --ICLR 2019

Meta-Learning for Semi-Supervised Few-Shot Classification [[paper](https://arxiv.org/pdf/1803.00676.pdf)]
  - Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richard S. Zemel --ICLR 2018

Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace [[paper](https://arxiv.org/abs/1903.08254)]
  - Kate Rakelly, Aurick Zhou, Deirdre Quillen, Chelsea Finn, Sergey Levine  --ICML 2018

### Self-supervised learning

MAML is a Noisy Contrastive Learner in Classification [[paper](https://openreview.net/pdf?id=LDAwu17QaJz)]
  - Chia Hsiang Kao, Wei-Chen Chiu, Pin-Yu Chen --ICLR 2022

Contrastive Learning is Just Meta-Learning [[paper](https://openreview.net/forum?id=gICys3ITSmj)]
  - Renkun Ni, Manli Shu, Hossein Souri, Micah Goldblum, Tom Goldstein --ICLR 2022

### Learning curves
Transferring Knowledge across Learning Processes [[paper](https://openreview.net/forum?id=HygBZnRctX)]
  - Sebastian Flennerhag, Pablo G. Moreno, Neil D. Lawrence, Andreas Damianou --ICLR 2019

Meta-Curvature [[paper](https://arxiv.org/abs/1902.03356)]
  - Eunbyung Park, Junier B. Oliva --NeurIPS 2019

### Hyperparameter
LCC: Learning to Customize and Combine Neural Networks for Few-Shot Learning [[paper](https://arxiv.org/pdf/1904.08479.pdf)]
  - Yaoyao Liu, Qianru Sun, An-An Liu, Yuting Su, Bernt Schiele, Tat-Seng Chua --CVPR 2019

Gradient-based Hyperparameter Optimization through Reversible Learning [[paper](https://arxiv.org/pdf/1502.03492.pdf)]
  - Dougal Maclaurin, David Duvenaud, Ryan P. Adams --ICML 2016

## Model compression
N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning
  - Anubhav Ashok, Nicholas Rhinehart, Fares Beainy, Kris M. Kitani --ICLR 2018

## Kernel learning

Deep Kernel Transfer in Gaussian Processes for Few-shot Learning [[paper](https://arxiv.org/pdf/1910.05199.pdf)]
  - Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O’Boyle, Amos Storkey --arXiv 2020

Deep Mean Functions for Meta-Learning in Gaussian Processes [[paper](https://arxiv.org/pdf/1901.08098.pdf)]
  - Vincent Fortuin, Gunnar Rätsch --arXiv 2019

Kernel Learning and Meta Kernels for Transfer Learning  [[paper](http://www1.icsi.berkeley.edu/~rueckert/papers/rueckert09kernel)]
  - Ulrich Ruckert

## Robustness
A Closer Look at the Training Strategy for Modern Meta-Learning [[paper](https://proceedings.neurips.cc/paper/2020/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html)]
  - JIAXIN CHEN, Xiao-Ming Wu, Yanke Li, Qimai LI, Li-Ming Zhan, Fu-lai Chung --NeurIPS 2020

Task-Robust Model-Agnostic Meta-Learning [[paper](https://papers.nips.cc/paper/2020/file/da8ce53cf0240070ce6c69c48cd588ee-Paper.pdf)]
  - Liam Collins, Aryan Mokhtari, Sanjay Shakkottai --NeurIPS 2020

FeatureBoost: A Meta-Learning Algorithm that Improves Model Robustness [[paper](https://hunch.net/~jl/projects/robust/ml2krobust.pdf)]
  - Joseph O'Sullivan, John Langford, Rich Caruana, Avrim Blum --ICML 2000

## Optimization

Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning [[paper](https://arxiv.org/abs/2206.03996)]
  - Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, Tianyi Chen --ICML 2022

Bootstrapped Meta-Learning [[paper](https://openreview.net/forum?id=b-ny3x071E5)]
  - Sebastian Flennerhag, Yannick Schroecker, Tom Zahavy, Hado van Hasselt, David Silver, Satinder Singh --ICLR 2022

Learning where to learn: Gradient sparsity in meta and continual learning [[paper](https://arxiv.org/pdf/2110.14402.pdf)]
  - Johannes von Oswald, Dominic Zhao, Seijin Kobayashi, Simon Schug, Massimo Caccia, Nicolas Zucchet, João Sacramento --NeurIPS 2021


Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML [[paper](https://openreview.net/pdf/f0530e2cf88af3b74bf61bc8591b7a5a1339c49e.pdf)]
  - Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals --ICLR 2020

Empirical Bayes Transductive Meta-Learning with Synthetic Gradients [[paper](https://arxiv.org/abs/2004.12696)]
  - Shell Xu Hu, Pablo G. Moreno, Yang Xiao, Xi Shen, Guillaume Obozinski, Neil D. Lawrence, Andreas Damianou --ICLR 2020

Transferring Knowledge across Learning Processes [[paper](https://openreview.net/forum?id=HygBZnRctX)]
  - Sebastian Flennerhag, Pablo G. Moreno, Neil D. Lawrence, Andreas Damianou --ICLR 2019

MetaInit: Initializing learning by learning to initialize [[paper](https://papers.nips.cc/paper/9427-metainit-initializing-learning-by-learning-to-initialize)]
  - Yann N. Dauphin, Samuel Schoenholz  --NeurIPS 2019
  
Meta-Learning with Implicit Gradients [[paper](https://arxiv.org/abs/1909.04630)]
  - Aravind Rajeswaran*, Chelsea Finn*, Sham Kakade, Sergey Levine  --NeurIPS 2019

Model-Agnostic Meta-Learning using Runge-Kutta Methods [[paper](https://arxiv.org/abs/1910.07368)]
  - Daniel Jiwoong Im, Yibo Jiang, Nakul Verma --arXiv

Learning to Optimize in Swarms [[paper](https://arxiv.org/pdf/1911.03787.pdf)]
  - Yue Cao, Tianlong Chen, Zhangyang Wang, Yang Shen --arXiv 2019

Meta-Learning with Warped Gradient Descent [[paper](https://arxiv.org/pdf/1909.00025.pdf)]
  - Sebastian Flennerhag, Andrei A. Rusu, Razvan Pascanu, Hujun Yin, Raia Hadsell --ICLR 2020
  
Learning to Generalize to Unseen Tasks with Bilevel Optimization [[paper](https://arxiv.org/pdf/1908.01457.pdf)]
  - Hayeon Lee, Donghyun Na, Hae Beom Lee, Sung Ju Hwang --arXiv 2019

Learning to Optimize [[paper](https://arxiv.org/abs/1606.01885)]
  - Ke Li Jitendra Malik --ICLR 2017

Gradient-based Hyperparameter Optimization through Reversible Learning [[paper](https://arxiv.org/pdf/1502.03492.pdf)]
  - Dougal Maclaurin, David Duvenaud, Ryan P. Adams --ICML 2016

### Continuous time

Continuous-Time Meta-Learning with Forward Mode Differentiation [[paper](https://openreview.net/forum?id=57PipS27Km)]
  - Tristan Deleu, David Kanaa, Leo Feng, Giancarlo Kerg, Yoshua Bengio, Guillaume Lajoie, Pierre-Luc Bacon --ICLR 2022

Meta-learning using privileged information for dynamics [[paper](https://arxiv.org/pdf/2104.14290.pdf)]
  - Ben Day, Alexander Norcliffe, Jacob Moss, Pietro Liò --ICLR 2020 #Learning to Learn and SimDL

## Theory

Near-Optimal Task Selection with Mutual Information for Meta-Learning [[paper](https://www.comp.nus.edu.sg/~lowkh/pubs/aistats2022.pdf)]
  - Chen, Yizhou; Zhang, Shizhuo; Low, Bryan Kian Hsiang  --AISTATS 2022

Learning Tensor Representations for Meta-Learning [[paper](https://arxiv.org/abs/2201.07348)]
  - Samuel Deng, Yilin Guo, Daniel Hsu, Debmalya Mandal --AISTATS 2022

Is Bayesian Model-Agnostic Meta Learning Better than Model-Agnostic Meta Learning, Provably? [[paper](https://chentianyi1991.github.io/bamaml_aistats2022.pdf)]
  - Lisha Chen, Tianyi Chen  --AISTATS 2022

Unraveling Model-Agnostic Meta-Learning via The Adaptation Learning Rate [[paper](https://openreview.net/forum?id=3rULBvOJ8D2)]
  - Yingtian Zou, Fusheng Liu, Qianxiao Li --ICLR 2022

Task Relatedness-Based Generalization Bounds for Meta Learning [[paper](https://openreview.net/forum?id=A3HHaEdqAJL)]
  - Jiechao Guan, Zhiwu Lu --ICLR 2022

A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning [[paper](https://arxiv.org/pdf/2106.15615.pdf)]
  - Nikunj Saunshi, Arushi Gupta, and Wei Hu --ICML 2021

Bilevel Optimization: Convergence Analysis and Enhanced Design [[paper](http://proceedings.mlr.press/v139/ji21c/ji21c.pdf)]
  - Kaiyi Ji, Junjie Yang, Yingbin Liang --ICML 2021

How Important is the Train-Validation Split in Meta-Learning? [[paper](https://arxiv.org/abs/2010.05843)]
  - Yu Bai, Minshuo Chen, Pan Zhou, Tuo Zhao, Jason D. Lee, Sham Kakade, Huan Wang, Caiming Xiong --ICML 2021

Information-Theoretic Generalization Bounds for Meta-Learning and Applications [[paper](https://arxiv.org/pdf/2005.04372.pdf)]
  - Sharu Theresa Jose, Osvaldo Simeone --arXiv 2021


Modeling and Optimization Trade-off in Meta-learning [[paper](https://proceedings.neurips.cc/paper/2020/hash/7fc63ff01769c4fa7d9279e97e307829-Abstract.html)]
  - Katelyn Gao, Ozan Sener --NeurIPS 2020

A Closer Look at the Training Strategy for Modern Meta-Learning [[paper](https://proceedings.neurips.cc/paper/2020/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html)]
  - JIAXIN CHEN, Xiao-Ming Wu, Yanke Li, Qimai LI, Li-Ming Zhan, Fu-lai Chung --NeurIPS 2020

Why Does MAML Outperform ERM? An Optimization Perspective [[paper](https://arxiv.org/pdf/2010.14672.pdf)]
  - Liam Collins, Aryan Mokhtari, Sanjay Shakkottai --arXiv 2020

Transfer Meta-Learning: Information-Theoretic Bounds and Information Meta-Risk Minimization [[paper](https://arxiv.org/pdf/2011.02872.pdf)]
  - Sharu Theresa Jose, Osvaldo Simeone, Giuseppe Durisi --arXiv 2020

The Advantage of Conditional Meta-Learning for Biased Regularization and Fine-Tuning [[paper](https://arxiv.org/abs/2006.09486)]
  - Giulia Denevi, Massimiliano Pontil, Carlo Ciliberto --NeurIPS 2020

Convergence of Meta-Learning with Task-Specific Adaptation over Partial Parameters [[paper](https://arxiv.org/abs/2006.09486)]
  - Kaiyi Ji, Jason D. Lee, Yingbin Liang, H. Vincent Poor --NeurIPS 2020

Meta-learning for mixed linear regression [[paper](https://proceedings.icml.cc/static/paper_files/icml/2020/6124-Paper.pdf)]
  - Weihao Kong, Raghav Somani, Zhao Song, Sham Kakade, Sewoong Oh --ICML 2020

Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time
  - Ferran Alet, Kenji Kawaguchi, Maria Bauza, Nurallah Giray Kuru, Tomás Lozano-Pérez, Leslie Pack Kaelbling  --NeurIPS 2020 #Meta-Learning

A Theoretical Analysis of the Number of Shots in Few-Shot Learning [[paper](https://openreview.net/forum?id=HkgB2TNYPS)]
  - Tianshi Cao, Marc T Law, Sanja Fidler --ICLR 2020

Efficient Meta Learning via Minibatch Proximal Update [[paper](https://papers.nips.cc/paper/8432-efficient-meta-learning-via-minibatch-proximal-update)]
  - Pan Zhou, Xiaotong Yuan, Huan Xu, Shuicheng Yan, Jiashi Feng --NeurIPS 2019

On the Convergence Theory of Gradient-Based Model-Agnostic Meta-Learning Algorithms [[paper](https://arxiv.org/pdf/1908.10400.pdf)]
  - Alireza Fallah, Aryan Mokhtari, Asuman Ozdaglar --arXiv 2019

Meta-learners' learning dynamics are unlike learners' [[paper](https://arxiv.org/abs/1905.01320)]
  - Neil C. Rabinowitz --arXiv 2019

Regret bounds for meta Bayesian optimization with an unknown Gaussian process prior [[paper](https://arxiv.org/pdf/1811.09558.pdf)]
  - Zi Wang, Beomjoon Kim, Leslie Pack Kaelbling --NeurIPS 2018

Incremental Learning-to-Learn with Statistical Guarantees [[paper](https://arxiv.org/abs/1803.08089)]
  - Giulia Denevi, Carlo Ciliberto, Dimitris Stamos, Massimiliano Pontil  --UAI 2018

Meta-learning by adjusting priors based on extended PAC-Bayes theory [[paper](https://arxiv.org/pdf/1711.01244.pdf)]
  - Ron Amit , Ron Meir --ICML 2018

Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm [[paper](https://arxiv.org/pdf/1710.11622.pdf)]
  - Chelsea Finn, Sergey Levine --ICLR 2018

On the Convergence of Model-Agnostic Meta-Learning [[paper](http://noahgolmant.com/writings/maml.pdf)]
  - Noah Golmant

Fast Rates by Transferring from Auxiliary Hypotheses [[paper](https://arxiv.org/pdf/1412.1619.pdf)]
  - Ilja Kuzborskij, Francesco Orabona --arXiv 2014

Algorithmic Stability and Meta-Learning  [[paper](http://www.jmlr.org/papers/volume6/maurer05a/maurer05a.pdf)]
  - Andreas Maurer  --JMLR 2005


### Online convex optimization
PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees [[paper](http://proceedings.mlr.press/v139/rothfuss21a/rothfuss21a.pdf)]
  - Jonas Rothfuss, Vincent Fortuin, Martin Josifoski, Andreas Krause --ICML 2021

Meta-learning with Stochastic Linear Bandits [[paper](https://arxiv.org/abs/2005.04372)]
  - Leonardo Cella, Alessandro Lazaric, Massimiliano Pontil --arXiv 2020

Bayesian Online Meta-Learning with Laplace Approximation [[paper](https://arxiv.org/abs/2005.00146)]
  - Pau Ching Yap, Hippolyt Ritter, David Barber --arXiv 2020

Online Meta-Learning on Non-convex Setting [[paper](https://arxiv.org/abs/1910.10196)]
  - Zhenxun Zhuang, Yunlong Wang, Kezi Yu, Songtao Lu --arXiv 2019

Adaptive Gradient-Based Meta-Learning Methods [[paper](https://arxiv.org/pdf/1906.02717.pdf)]
  - Mikhail Khodak, Maria-Florina Balcan, Ameet Talwalkar --NeurIPS 2019

Learning-to-Learn Stochastic Gradient Descent with Biased Regularization [[paper](https://arxiv.org/abs/1903.10399)]
  - Giulia Denevi, Carlo Ciliberto, Riccardo Grazzi, Massimiliano Pontil  --NeurIPS 2019

Provable Guarantees for Gradient-Based Meta-Learning
  -  Mikhail Khodak Maria-Florina Balcan Ameet Talwalkar --arXiv 2019 

