# PhysicalMultiTimeSeries

## problem setup
As part of my thesis work, we aim to learn a generalized mapping from the lift force $F_{lift}(t)$ generated by a flapping wing to its $3$ degrees of angle motion pitch yaw and roll $\left(\phi(t), \psi(t), \theta(t)\right):=K(t)$.

Generalizing the above, given a time window of features with size $H$

$$[F_1(t),F_2(t),...,F_M(t)]_{t=t_0}^{t_0+H}$$

we provide a prediction up to time $t_0+H+T$
$$[T_1(t),T_2(t),...,T_N(t)]_{t=t_0+H+1}^{t_0+H+T}$$


## Framework
PMTS is a generalized library for multi target, multivariate time series prediction for physical experiments, mapping kinematics to forces or vice-versa.

We have developed a framework that offers a `Trainer` class, which can be used to train a customized nn.Module model from PyTorch. The Trainer class is designed to work with features tensor shaped as `(N,H,F)` and target tensor shaped as `(N,H,T)`, where `N` represents the number of datasets, `H` represents the provided history, `F` represents the number of features, and `T` represents the number of targets.

Our Trainer class comes with commonly used parameters such as batch size, train size, feature window size, and more. With an intuitive API similar to other popular frameworks, you can use the `.fit()` function to train your model and the `.predict()` function to generate predictions.

A working example is provided under `main.py`


## Recent Updates
We now support Normalizations, both for the features and the test taret. The normalization is perform in the pre-processing step of train-val-test split. Might still not support un-normalizing the predicted values yet.

## Working on
Please note that the following features are not yet available in our framework, but they are expected to be implemented in the near future:

- Pre-processing module: We plan to enhance our pre-processing capabilities by adding features such as normalizing, resampling, and interpolating data. This will improve data quality and reduce noise, resulting in better model performance.

- Encoders for feature engineering: We will expand our feature engineering capabilities by developing additional encoders like derivatives and frequeicies extractio to handle different types of features and improve model performance. 
