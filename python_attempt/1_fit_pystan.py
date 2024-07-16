## now fit a gaussian mixture to these datasets, using pystan

import numpy as np
import pystan
import pandas as pd

# define the stan model
model_code = """
data {
    int<lower=1> N; // number of data points
    int<lower=1> K; // number of components
    real y[N]; // data
}
parameters {
    simplex[K] theta; // mixing proportions
    real mu[K]; // component means
    real<lower=0> sigma[K]; // component std. deviations
}
model {
    real ps[K];
    real lp;
    for (n in 1:N) {
        for (k in 1:K) {
            ps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
        }
        lp = log_sum_exp(ps);
        target += lp;
    }
}
"""

# compile the model
model = pystan.StanModel(model_code=model_code)

# read in the generated data
pop_lambda = 1.0
pop_pcall = 0.1

fname = f"simulated_data/simulated_truthlambda_{pop_lambda:.2f}_p_call_{p_call:.2f}.csv"

# read in with pandas 

pd.read_csv(fname)

# fit the model
fit = model.sampling(data=data, iter=1000, chains=4)

# extract the results
results = fit.extract()
