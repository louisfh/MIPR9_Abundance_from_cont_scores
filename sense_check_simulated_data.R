## Sense check the simulated data
## by looking at the histograms

#files
save_dir <- "/Users/LOF19/Documents/kitzes_projects/Rotation_2_continuous_score/simulation_sunday/data/"

#make a figure with 3 rows and 2 columns
par(mfrow = c(4,4))
pdf(sprintf("%shists.pdf", save_dir))

# loop over the files
for (lambda in seq(0,4,0.2)){
  data_file <- sprintf("%ssimulated_scores%g.Rdata", save_dir, lambda)
  load(data_file)
  # plot the histogram
  hist(scores, main = sprintf("Lambda is %g \n p is %g", lambda, p_call[1]))
}
dev.off()
