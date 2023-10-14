# Fusion-CAM
**Fusion-CAM: Segmentation Pseudo-label Generation using the Multiple Instance Learning Choquet Integral**

_Connor McCurley_

In this repository, we provide a Python implementation of the Fusion-CAM algorithm.  

___
The assiciated papers for this repository are:

- [[`Gatorsense publications page (Fusion-CAM PhD Thesis)`]((https://faculty.eng.ufl.edu/machine-learning/2023/02/discriminative-feature-learning-with-imprecise-uncertain-and-ambiguous-data/))]

- [[`BibTeX`](#CitingFusionCAM)]


## Installation Prerequisites

This code uses standard anaconda libraries.

## Cloning

To recursively clone this repository using Git, use the following command:

```
git clone --recursive https://github.com/GatorSense/fusion-cam.git
```

<!---
## Demo

Run `demo_main.py` in Python.

## Main Functions

The MICI Classifier Fusion and Regression Algorithm runs using the following functions.

1. MICI Classifier Fusion (generalized-mean model) Algorithm

```train_chi_softmax(TrainBags, TrainBagLabels, Parameters)```


## Inputs

#The *TrainBags* input is a (NumTrainBags,) numpy object array. Inside each element, (NumPntsInBag, nSources) numpy array -- Training bags data.

#The *TrainLabels* input is a (NumTrainBags,) numpy uint8 array that takes values of "1" and "0" for two-class classfication problems -- Training labels for each bag.


## Parameters
The parameters can be set in the following function:

```Parameters = set_mici_parameters.set_parameters()```
```
The parameters are in a Python argument parser with the following fields:
1. nPop: size of population
2. sigma: sigma of Gaussians in fitness function
3. maxIterations: maximum number of iterations
4. eta: percentage of time to make small-scale mutation
5. sampleVar: variance around sample mean
6. mean: mean of CI in fitness function. This value is always set to 1 (or very close to 1) if the positive label is "1".
7. analysis: if ="1", save all intermediate results
8. p: the power coefficient for the generalized-mean function. Empirically, setting p(1) to a large postive number and p(2) to a large negative number works well.
9. use_parallel: flag for using parallel processing in evolutionary optimization sampling.
```

*Parameters can be modified by users in Parameters = set_parameters() function.*

## Inventory

```
https://github.com/GatorSense/fusion-cam

└── root dir
    ├── mici_demo_main.py   //Run this. Main demo file.
    ├── demo_data_cl.mat  //Demo classification data
    ├── mici_choquet_integral.py  //Class file for the MICI with regular fuzzy measure
    ├── mici_choquet_integral_binary.py  //Class file for the MICI with binary fuzzy measure
    ├── set_mici_parameters.py  //Parameter file
```
-->

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2022 C.McCurley, and A. Zare. All rights reserved.

## <a name="CitingFusionCAM"></a>Citing Fusion-CAM

If you use the Fusion-CAM algorithm, please cite the following references using the following entry.

__Plain Text:__

C. McCurley, "Discriminatve Feature Learning with Imprecise, Uncertain, and Ambiguous Data," Ph.D Thesis, Gainesville, FL, 2022.


__BibTex:__
```
@phdthesis{mccurley2022thesis,
author={C. McCurley},
title={Discriminative Feature Learning with Imprecise, Uncertain, and Ambiguous Data},
school={Univ. of Florida},
year={2022},
address={Gainesville, FL},
}
```

## Related Work

Also check out our Multiple Instance Choquet Integral (MICI) algorithm for information fusion!

[[`IEEE Explore`](https://ieeexplore.ieee.org/document/7743905)] 

[[`GitHub Code Repository`](https://github.com/GatorSense/MICI)]

## Further Questions

For any questions, please contact:

Alina Zare

Email Address: azare@ece.ufl.edu

University of Florida, Department of Electrical and Computer Engineering

