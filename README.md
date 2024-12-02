# DCL: An Efficient and Scalable Method for Customer Targeting in Market Data Management

### ***UPDATED ON 2024.12.02 ***  


### ***Directory Structure***
The code to replicate the offline results in **Section IV. EXPERIMENTS --> A. Offline Test**
```
|----- code_Criteo       # Various benchmark methods when the dataset is CRITEO-UPLIFT v2.
|----- code_MT           # Various benchmark methods when the dataset is Meituan-LIFT.
|----- code_Ali          # Various benchmark methods when the dataset is Alibaba-LIFT.
|----- metric
|     |-----Metric.py               # The evaluation metrics: AUCC
|-----model
|     |-----uplift_model.py         # The model to predict CATE
|     |-----roi_model.py            # The model to predict ROI
```


### ***Three Real-world Public Industrial Dataset***
```
1a. Dataset name: CRITEO-UPLIFT v2
    Download link: https://ailab.criteo.com/criteo-uplift-prediction-dataset/, rename it as "criteo-uplift-v2.1.csv"
1b. Dataset name: Meituan-LIFT
    Download link: GitHub - MTDJDSP/MT-LIFT: The real-world Production Dataset from E-commerce platform Meituan, rename it as "/MT-LIFT/train.csv"
1c. Dataset name: Alibaba-LIFT
    Download link: https://tianchi.aliyun.com/dataset/94883, rename it as "Alibaba-lift.csv"
2. Make a "data" directory, and put these three datasets in the data directory.
```


### ***Setup Details***
```
1. Tensorflow 2.14.0 is used in this experiment.
2. Neural network ran on a machine with a GPU RTX 4090(24GB) and 90GB memory.
3. To support the use of Generalized Random Forests (GRF) , install econML from GitHub - py-why/EconML: ALICE (Automated Learning and Intelligence for Causation and Economics) is a.
4. GRF ran on a machine with 32 vCPU (AMD EPYC 7742 64-Core Processor) and 96GB memory.
```
