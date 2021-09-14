# MCNet

![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

## Document description

```
---| train_dataset
---| eval_dataset1
---| eval_dataset2
---| eval_dataset3
---| save_result_dataset1  # Results folder for the first test dataset
---| save_result_dataset2  # Results folder for the second test dataset
---| save_result_dataset3  # Results folder for the third test dataset
---| models  # Folder for storing various neural networks
---| utils # Folder for storing various scripts
---| selectBest.py # Select the best result from the result dir
---| run.py  # the start script of training
---| run_eval.py # the start script of evaling
---| infer_time.py # the script for calculate the infer speed of network
---| config.yml # configuration file
```

