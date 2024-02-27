This directory contains work on MIPR9. This is an extension to Tessa's continuous-score occupancy model.

I am going to extend the model to abundance, by assuming that there is a poisson distributed number of birds at any given site (with intensity lambda)

Their song is a poisson process. They sing in  certain number of time-windows.

>> write out the data

score_t ~ gaussian_mixture(mu1, mu2, sigma1, sigma2, theta)
(we have N_clips score_ts)

Number_of_time_windows_not_containing_song = N_clips*(1-theta)




N ~ Poisson(lambda) # the number of birds at a site

