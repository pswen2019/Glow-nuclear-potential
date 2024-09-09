# Glow for nuclear potentials
This project applies Glow model to build nuclear potentials. 

## 1. Generate chiral potential samples 
The chiral potential sample generators are in the directory `chiralsample/chiraln2lo` and `chiralsample/chiraln3lo`.
Use the following cmd to generate potential sampels whose LECs following normal distributions:
```bash
python multiprocess_pot_generate_normal_distribution_for_LEC.py --numprocess 10
```
The above cmd will use 10 independent processes to generate 300 samples totally 
for cutoff $\Lambda = 450, 500$ and $550$ MeV in directory `allpot`. 
Each cutoff value has 100 potential samples. 
A file named with '.dat' records potential matrix elements.
The corresponding LECs and $\Lambda$ are records in file named with '.txt'

## 2. Train a Glow model
Train a Glow model maximizing likelihood
```
python train.py --datasetdir path_to_potential_samples_directory
```
IF the trained ViT models are provided, 
`train.py` will train a Glow model based on likelihood and $|\Lambda_{\rm Glow} - \Lambda_{\rm ViT}|$
The cmd for this combined training can be found in `runtrain.sh`

## 3. Generate potentials using trained Glow model
`predict.py` is used to generate potentials with a trained Glow model. 
```bash
python predict.py --model path_to_trained_Glow_model --modelconfig path_to_configuration
```
The above cmd will load a trained Glow model and generate chiral n2 and n3 potentials with cutoff 
$$
\Lambda \in \{450, 460, 470, ..., 550\} {\rm MeV}
$$
If trained ViT models are provided, 
the ViT models will extract LECs and $\Lambda$ from the generated potentials. 
The method to use ViT model can be found in `runpredict.sh`.
