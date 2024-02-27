# %%
# I will inspect some histograms, in order to work out reasonable mu1, mu2, sigma1, sigma2 values for the mixture model
# I will start by looking at Cam's WOTH data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
preds = pd.read_csv('./real_data/sgl60_cam_woth.csv', index_col=[0,1,2])
preds

# %%
all_data = preds['WOTH_song_x'].values
# fit a gaussian mixture with 2 components to this data
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(all_data.reshape(-1,1))
print(f"Means found: {gmm.means_}")
print(f"Covariances found: {gmm.covariances_}")
print(f"Weights found: {gmm.weights_}")

# %%
# plot the two gaussian
# I will use the means and covariances found by the GMM
import math
# use large plots rcparams
plt.rcParams['figure.figsize'] = [15, 10]

mu1 = gmm.means_[0][0]
mu2 = gmm.means_[1][0]
sigma1 = math.sqrt(gmm.covariances_[0][0])
sigma2 = math.sqrt(gmm.covariances_[1][0])
x = np.linspace(min(all_data), max(all_data), 300)
y1 = np.exp(-0.5*((x-mu1)/sigma1)**2)/(sigma1*np.sqrt(2*np.pi)) * gmm.weights_[0]
y2 = np.exp(-0.5*((x-mu2)/sigma2)**2)/(sigma2*np.sqrt(2*np.pi)) * gmm.weights_[1]
plt.plot(x, y1, label='Gaussian 1')
plt.plot(x, y2, label='Gaussian 2')
plt.hist(all_data, bins=500, density=True, alpha=0.5)

# add a line for the sum of the two gaussians
y = y1 + y2
plt.plot(x, y, label='Sum of Gaussians', linestyle='--')

plt.legend()

# %%
def fit_n_gaussians_and_plot(data, n_components, n_hist_bins, title='Two Component Gaussian Mixture Model', figsize=(15, 10), log=False):
    """
    Fit an n_component Gaussian Mixture Model to the data y and plot the result
    Args:
        data: np.array
        n_components: int
        n_hist_bins: int
        title: str
        figsize: tuple
        log: bool. Whether to fit the normals, or log normals. If True, the histogram will be log transformed before fitting the model
    Returns:
        gmm: sklearn.mixture.GaussianMixture
        ax: matplotlib.axes._subplots.AxesSubplot
    """

    from sklearn.mixture import GaussianMixture
    import math
    
    fig, ax = plt.subplots(figsize=figsize)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(data.reshape(-1,1))

    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    
    x = np.linspace(min(data), max(data), n_hist_bins)

    sum_of_gaussians = np.zeros(n_hist_bins)
    for i in range(n_components):
        mu = means[i][0]
        sigma = math.sqrt(covariances[i][0])
        y = np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)) * weights[i]
        sum_of_gaussians += y
        ax.plot(x, y, label=f"Gaussian {i+1} μ: {mu:.2f}, σ: {sigma:.2f}, θ: {weights[i]:.2f}", linestyle='-')
    
    # add a line for the sum of the gaussians
    ax.plot(x, sum_of_gaussians, label='Sum of Gaussians', linestyle='--', color='black')

    # add the observed data to the plot
    ax.hist(data, bins=n_hist_bins, density=True, alpha=0.2, color = 'green', label='Observed Data')
    
    ax.set_title(title)
    # label y axis
    ax.set_ylabel('Density')

    ax.legend()

    # set axis y limit
    return (gmm, ax)

fit_n_gaussians_and_plot(all_data, n_components=2, n_hist_bins=500, title='Two Component Gaussian Mixture Model on all SGL60 WOTH scores', figsize=(15, 10), log=False)

# %%
positives = preds[preds["WOTH_song_annotation"] == 1]
negatives = preds[preds["WOTH_song_annotation"] == 0]
unknowns = preds[preds["WOTH_song_annotation"].isna()]

# %%
len(positives), len(negatives), len(unknowns)

# %%
plt.hist(positives['WOTH_song_x'], bins=50, density=True, alpha=0.5, label='Positives')
plt.hist(negatives['WOTH_song_x'], bins=100, density=True, alpha=0.5, label='Negatives')
plt.hist(unknowns['WOTH_song_x'], bins=500, density=True, alpha=0.5, label='Unannotated')
plt.legend()
plt.title('Histogram of WOTH scores for SGL60. Top 1 per site, per day, from 3 time-windows were annotated')

# %%
# test the normality of the data using the Shapiro-Wilk test
from scipy.stats import shapiro
shapiro(all_data)

# %%
# try subsampling the data, and see how many samples you have to draw in order to fail the shapiro-wilk
subsample_sizes = np.arange(10, 500, 10)
num_repeats = 1000
p_values = np.zeros(len(subsample_sizes))
for i, n in enumerate(subsample_sizes):
    p_values[i] = np.mean([shapiro(np.random.choice(all_data, n))[1] for _ in range(num_repeats)])
plt.plot(subsample_sizes, p_values, label='Random subsample')

for i, n in enumerate(subsample_sizes):
    p_values[i] = np.mean([shapiro(np.random.choice(positives["WOTH_song_x"], n))[1] for _ in range(num_repeats)])
plt.plot(subsample_sizes, p_values, label='Positives')

for i, n in enumerate(subsample_sizes):
    p_values[i] = np.mean([shapiro(np.random.choice(negatives["WOTH_song_x"], n))[1] for _ in range(num_repeats)])
plt.plot(subsample_sizes, p_values, label='Negatives')

plt.legend()
plt.xlabel('Number of samples')
plt.ylabel('Mean p-value from Shapiro-Wilk test')
plt.title(f'Shapiro-Wilk test p-value vs number of samples randomly chosen from sgl_60 WOTH scores ({num_repeats} repeats per sample size)')

# %%
# maybe I can try quantifying distribution shift by picking a relatively large number of samples, then calculating the KL divergence between the distributions of the samples
def subset_by_method(all_data, data_to_match, method, **kwargs):
    """
    A function for subsetting data, according to the method specified
    Args:
        all_data: pandas.DataFrame, containing all the data, and the columns to subset by
        data_to_match: pandas.DataFrame, containing the data that you're going to match. This data should not be present in all_data 
                        (to avoid getting the same data back when querying all_data). 
        method: str. One of "site", "date", "time_period", "site-date", "site-time_period", "date-time_period", "site-date-time_period"
    Returns:
        subset: pandas.DataFrame
    """

    matching_site = data_to_match["site"].sample(1).values[0]
    matching_date = data_to_match["date"].sample(1).values[0]
    matching_time_period = data_to_match["time_period"].sample(1).values[0]

    # check that data_to_match is not present in all_data
    # do this by checking the index
    assert data_to_match.index.isin(all_data.index).sum() == 0, "The data to match should not be present in all_data"
    
    if method == "site":
        assert data_to_match["site"].nunique() == 1, "The data to match should only contain data from one site"
        subset = all_data[all_data["site"] == matching_site]
    elif method == "date":
        assert data_to_match["date"].nunique() == 1, "The data to match should only contain data from one day"
        subset = all_data[all_data["date"] == random_day]
    elif method == "time_period":
        assert data_to_match["time_period"].nunique() == 1, "The data to match should only contain data from one time period"
        subset = all_data[all_data["time_period"] == matching_time_period]
    elif method == "site-date":
        assert data_to_match["site"].nunique() == 1, "The data to match should only contain data from one site"
        subset = all_data[(all_data["site"] == matching_site) & (all_data["date"] == matching_date)]
    elif method == "site-time_period":
        assert data_to_match["site"].nunique() == 1, "The data to match should only contain data from one site"
        assert data_to_match["time_period"].nunique() == 1, "The data to match should only contain data from one time period"
        subset = all_data[(all_data["site"] == matching_site) & (all_data["time_period"] == matching_time_period)]
    elif method == "date-time_period":
        assert data_to_match["date"].nunique() == 1, "The data to match should only contain data from one day"
        assert data_to_match["time_period"].nunique() == 1, "The data to match should only contain data from one time period"
        subset = all_data[(all_data["date"] == matching_date) & (all_data["time_period"] == matching_time_period)]          
    elif method == "site-date-time_period":
        assert data_to_match["site"].nunique() == 1, "The data to match should only contain data from one site"
        assert data_to_match["date"].nunique() == 1, "The data to match should only contain data from one day"
        assert data_to_match["time_period"].nunique() == 1, "The data to match should only contain data from one time period"
        subset = all_data[(all_data["site"] == matching_site) & (all_data["date"] == matching_date) & (all_data["time_period"] == matching_time_period)]
    else:
        raise ValueError(f"Method {method} not recognised")
    
    return subset

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

# %%
# drop the time_periods that are NAs
only_complete_time_periods = preds.dropna(subset=["time_period"])

# %%
# pick a random site, date and time_period
num_iterations = 1000
kl_divergences = {}
reverse_kl_divergences = {}

# get every site-date-time_period combination that has more than 400 samples
possible_date_time_periods = only_complete_time_periods.groupby(["site", "date", "time_period"]).size().reset_index()
possible_date_time_periods = possible_date_time_periods[possible_date_time_periods[0] > 400]

for subsetting_method in ["site", "date", "time_period", "site-date", "site-time_period", "date-time_period", "site-date-time_period"]:

    divergences = []
    reverse_divergences = []
    
    for i in range(num_iterations):
        # pick a random site, date and time_period from the possible combinations
        row = possible_date_time_periods.sample(1)
        random_site = row["site"].values[0]
        random_day = row["date"].values[0]
        random_time_period = row["time_period"].values[0]

        # get all the data that matches the random site, date and time_period
        all_data_to_match = only_complete_time_periods[(only_complete_time_periods["site"] == random_site) & (only_complete_time_periods["date"] == random_day) & (only_complete_time_periods["time_period"] == random_time_period)]
        
        # sample some of that data
        data_to_match = all_data_to_match.sample(200, replace = False) # half the size of the average site-date-time_period subset

        # drop those samples from the remaining data
        remaining_data = only_complete_time_periods.drop(data_to_match.index) # drop the data that was sampled

        # subset the remaining data to find those that match, according to the subsetting_method
        subset = subset_by_method(remaining_data, data_to_match, subsetting_method)
        print(f"{subsetting_method}: {len(subset)} samples")
        divergences.append(KLdivergence(data_to_match["WOTH_song_x"].values.reshape(-1, 1), subset["WOTH_song_x"].values.reshape(-1, 1)))
        reverse_divergences.append(KLdivergence(subset["WOTH_song_x"].values.reshape(-1, 1),data_to_match["WOTH_song_x"].values.reshape(-1, 1)))
        
    reverse_kl_divergences[subsetting_method] = reverse_divergences
    kl_divergences[subsetting_method] = divergences

# %%
# save the kl_divergences
kl_divergences = pd.DataFrame(kl_divergences)
kl_divergences.to_csv("CAM_sgl60_WOTH_kl_divergences.csv")

reverse_kl_divergences = pd.DataFrame(reverse_kl_divergences)
reverse_kl_divergences.to_csv("CAM_sgl60_WOTH_reverse_kl_divergences.csv")

# %%
# iterate through each method and calculate the means.  I will treat every -inf as a NaN, and just ignore it for the sake of mean calculation
means = {}
for method, kl_divergence in kl_divergences.items():
    kl_divergence = np.array(kl_divergence)
    kl_divergence[kl_divergence == -np.inf] = np.nan
    print(f"{method}: {np.nanmean(kl_divergence)}")
    means[method] = np.nanmean(kl_divergence)

# %%
# big plot rcparams
plt.rcParams['figure.figsize'] = [15, 10]
plt.bar(means.keys(), means.values())

# %%



