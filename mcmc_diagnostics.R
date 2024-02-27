### MCMC diagnostics ###
library(nimble)
library(tidyverse) # I think tidyverse required
library(rlist)
library(bayesplot)

save_dir <- "/Users/LOF19/Documents/kitzes_projects/Rotation_2_continuous_score/simulation_sunday/mcmc_results/"

#for combo plots
params_to_plot <- c("mu[1]", "mu[2]", "sigma[1]", "sigma[2]", "lambda")

for (real_lambda in seq(0,4,0.2)){
  results_filename <- sprintf("%sresults_%g.Rdata",save_dir, real_lambda)
  load(results_filename)
  
  pdf(sprintf("%scombo_plot_%g.pdf", save_dir, real_lambda))
  plot(mcmc_combo(continuous_results$samples, pars = params_to_plot))
  dev.off()
}