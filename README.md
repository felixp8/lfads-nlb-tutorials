# Modeling neural population activity and evaluating model fit with LFADS and NLB
Tutorial notebooks on LFADS and NLB for NeuroDataReHack 2022.

## Contents
This series of tutorials covers what LFADS is, how to run it, and how to evaluate its (and similar models') performance on real data. THe tutorials consist of three notebooks:
1. [Understanding LFADS](https://github.com/felixp8/lfads-nlb-tutorials/blob/main/understanding_lfads.ipynb) - core concepts underlying LFADS architecture
2. [Running LFADS](https://github.com/felixp8/lfads-nlb-tutorials/blob/main/running_lfads.ipynb) - training and tuning LFADS models using [autolfads-tf2](https://github.com/snel-repo/autolfads-tf2)
3. [Evaluating LFADS with NLB'21](https://github.com/felixp8/lfads-nlb-tutorials/blob/main/lfads_for_nlb.ipynb) - approaches for evaluating LFADS performance using Neural Latents Benchmark '21

All notebooks can be run in Google Colab (by clicking the "Open in Colab" button at the top of each notebook in GitHub) or run locally.

## Requirements
All notebooks require:
* Python >= 3.7
* numpy
* matplotlib
* scipy
* h5py

In addition, specific notebooks have particular dependencies that can be installed if needed from the notebook itself.

## Acknowledgements
The first two notebooks are mostly copied from [this tutorial](https://colab.research.google.com/drive/193q03zXj5379n5XnUQTyWkvt6oH2Xxmp?usp=sharing) by Mattia Rigotti and [this tutorial](https://colab.research.google.com/drive/1gJekKEzJTCiJZZXgzTPTxT_O0rAAf9tr?usp=sharing) by Lahiru Wimalasena, originally created for the [2021 Simons-Emory Theory Methods Workshop](https://www.internationalmotorcontrol.org/theoryworkshop.html). The third notebook is based on [this tutorial](https://github.com/neurallatents/nlb_workshop/blob/main/nlb_technical/nlb_technical_walkthrough.ipynb) from the NLB'21 Workshop.


