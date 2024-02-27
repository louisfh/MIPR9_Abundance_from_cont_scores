#### simulate data to test my MCMC model
## Author: Louis Freeland-Haynes
## Date: 2023/01/23

save_dir <- "/Users/LOF19/Documents/kitzes_projects/Rotation_2_continuous_score/simulation_sunday/data/"
for (lambda in seq(0,4,0.2)) {
  #### setup the simulation details ####
  NSITES <- 100
  NFILES <- 720
  
  ##### Parameters we want the model to discover #####
  # beta distribution parameters for call rate
  #alpha <- 1.1
  #beta <- 8.9
  #mean_p_call_rate <- alpha / (alpha+beta)
  
  # positive normal params
  positive_mu = 5
  positive_sd = 2
  # negative normal params
  negative_mu = -2
  negative_sd = 1
  
  ###### set up vectors for storing the results #####
  true_z <- vector() # occupancy
  abundance <- vector()
  p_call <- vector()
  p_at_least_1_singer <- vector() # is (1 - no birds sing at all)
  p_no_singers <- vector() # probability no bird sings in each period
  num_positives <- vector() # number of files that are from positives
  num_negatives <- vector() # number of files that are from positives
  scores <- matrix(NA, nrow = NSITES, ncol=NFILES)
  theta <- vector()
  z_data <- rep(NA, NSITES)# the annotations
  annotation_all <- matrix(NA, nrow = NSITES, ncol = NFILES)
  
  ###### loop over each site to make the data ####
  for (i in 1:NSITES) {
    #true_z[i] <- rbernoulli(1,psi)
    abundance[i] <- rpois(1, lambda)
    
    #p_call[i] <- rbeta(1, alpha, beta) 
    p_call[i] <- 0.25
    
    p_at_least_1_singer[i] <- (1-(1-p_call[i])**abundance[i])
    num_positives[i] <- rbinom(1, NFILES, p_at_least_1_singer[i])
    num_negatives[i] <- NFILES - num_positives[i]
    theta[i] <- num_positives[i]/NFILES
    scores[i,] <- c(rnorm(num_positives[i], positive_mu, positive_sd), rnorm(num_negatives[i], negative_mu, negative_sd))
  }
  
  filename = sprintf("%ssimulated_scores%g.csv", save_dir, lambda)
  write.csv(scores, file=filename)
  
  # write the rdata file with all the variables
  
  # save the data
  rdata_fname = sprintf("%ssimulated_scores%g.Rdata", save_dir, lambda)
  save(list = c("NSITES", "NFILES", "p_call", "positive_mu", "positive_sd", "negative_mu", "negative_sd","lambda", "annotation_all", "scores"), file = rdata_fname)
}

