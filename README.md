# PrivGGAN

This repository contains the implementation for paper PrivGGAN: A graph generation model with link differential privacy


Contact: Xiang Qiu ([xiangqiu@seu.edu.cn](mailto:xiangqiu@seu.edu.cn))


## Requirements 
The environment can be set up using [Anaconda](https://www.anaconda.com/download/) with the following commands:

``` setup
conda create --name privggan-pytorch python=3.6
conda activate privggan-pytorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
pip install -r requirements.txt
```

<!---Please note that modifications in registering the `backward_hook` (in `source/main.py`) may be required if you plan to use a different pytorch version. Please refer to the [pytorch document](https://pytorch.org/docs/versions.html) (select pytorch version &rarr; `torch.nn` &rarr;  `Module` &rarr; search for `register_backward_hook`) for more information.---> 

## Training 
#### Step 1. To warm-start the discriminators:
```warm-start
python pretrain.py --dataset_name 'citeseer'
```
- The above command pretrain the discriminators for 'citeseer' dataset (with 2000 iterations in default).
   
#### Step 2. To train the differentially private generator:
```train
python main.py --lr=1e-4 --dp_method="PrivGGAN"  --dataset_name='citeseer'
```
- Please refer to `config.py` (or execute `python main.py -h`) for the complete list of arguments. 

## Evaluation
#### Privacy
- To compute the privacy cost:
    ```privacy 
    cd evaluation
    python privacy_analysis.py 
    ```

#### Utility
- To evaluate the graph statistics: 
    ```statistics
    python evaluation.py --dataset 'citeseer' --model 'PrivGGAN' --target_epsilon 3.0
    ``` 
       
- To draw the degree distributions, one can use the following scripts
    ```degree distribution
      python draw_graph.py --dataset 'citeseer' --model 'PrivGGAN' --target_epsilon 3.0
    ```

<!-- 
## Citation
```bibtex

```
-->

## Acknowledgements

Our implementation uses the source code from the following repositories:

* [GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators (Pytorch)](https://github.com/DingfanChen/GS-WGAN)

* [NetGAN: Generating Graphs via Random Walks (Tensorflow)](https://github.com/danielzuegner/netgan)

* [NetGAN: Generating Graphs via Random Walks (Pytorch)](https://github.com/mmiller96/netgan_pytorch)
