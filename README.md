# Only Consider Persuadables and Coupon Takers? Cost-effective Resource Allocation via Duality Causal Learner


### ***Directory Structure***
The code to replicate the offline results in **Section 5. EXPERIMENTS --> 5.1. Offline Simulation**
```
|-----code
|     |-----【TPM-SL】.ipynb       # TPM-SL method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【TPM-XL】.ipynb       # TPM-XL method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【TPM-CF】.ipynb       # TPM-CF method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【DRM】.ipynb          # DRM method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【DRP】.ipynb          # DRP method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【DCL】.ipynb          # proposed DCL method AUCC results for C-BTAP when the sample ratios are from 0.11 to 0.21.
|     |-----【Fig.2】.ipynb        # Notebook codes that generate Fig.2: mean of offline AUCC results in eleven different sample ratios.
|-----figure
|     |-----sigir
|     |     |-----xxx.pdf           # The images result of 【Fig.2】.ipynb
|     |     |-----xxx.csv           # The AUCC result of different method in the code directory.
|-----metric
|     |-----Metric.py               # The evaluation metrics: AUCC
|-----model
|     |-----uplift_model.py         # The model to predict CATE
|     |-----roi_model.py            # The model to predict ROI
|-----model_file
|     |-----xxx                     # used to save the model files trained by various benchmark methods.       
|-----README.txt
```


### ***Public Dataset***
```
1. Dataset name: CRITEO-UPLIFT v2
2. Download link: https://ailab.criteo.com/criteo-uplift-prediction-dataset/, rename it as "criteo-uplift-v2.1.csv"
3. Make a "data" directory, and put this dataset in the data directory.
```


### ***Setup Details***
```
1. Tensorflow 2.14.0 is used in this experiment.
2. Neural network ran on a machine with a GPU GTX 1080 Ti 11GB and 30GB memory.
3. To support the use of Generalized Random Forests (GRF) , install econML from https://github.com/microsoft/EconML.
4. GRF ran on a machine with 7 vCPU (Xeon(R) E5-2680 v4) and 30GB memory.

```


