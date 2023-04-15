# MultiTimeSeries

### basic info
MTS is a generalized library for multi target multivariate time series prediction for physical experiments, 
mapping kinematics to forces or vice-versa.

### problem setup
As part of my thesis work, we aim to learn a generalized mapping from the lift force $F_{lift}(t)$ generated by a flapping wing to its 3 degrees of angle motion pitch yaw and roll $\left(\phi(t), \psi(t), \theta(t)\right):=K(t)$.

Generalizing the above, given a time window of features with size $H$

$$[F_1(t),F_2(t),...,F_M(t)]_{t=t_0}^{t_0+H}$$

we provide a prediction up to time $t_0+H+T$
$$[T_1(t),T_2(t),...,T_N(t)]_{t=t_0+H+1}^{t_0+H+T}$$


