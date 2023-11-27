import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import statsmodels.api as sm
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt


def best_fit_distribution(data, bins=200, ax=None):
    # https://stackoverflow.com/a/37616966
    """Model data by finding best fit distribution to data"""
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distributions = []

    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):
        distribution = getattr(st, distribution)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                params = distribution.fit(data)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    return sorted(best_distributions, key=lambda x:x[2])


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.DataFrame({
        'value': x,
        'proba': y
    })

    return pdf


def normalize_proba(df, min_value=None, max_value=None):
    if min_value is not None:
        df = df[df['value'] >= min_value]
    if max_value is not None:
        df = df[df['value'] <= max_value]
        
    df['proba'] = df['proba'] / df['proba'].sum()
    
    if(pd.isna(df['proba']).any()):
        df['proba'] = 1 / len(df['proba'])
        
    return df


def arr2hist(lst):
    hist, bin_edges = np.histogram(lst, density=True, bins='auto')
    step = bin_edges[1] - bin_edges[0]
    
    pdf = pd.DataFrame({
        'value': [(bin_edges[0] + (step / 2)) + (i * step) for i in range(len(bin_edges) - 1)],
        'proba': hist
    })
    
    pdf = normalize_proba(pdf)
    
    return pdf


def save_pdf(df, stat_name='default'):
    df.to_csv('tmp_data' + stat_name + '.csv', index=False)
    

def load_pdf(stat_name='default'):
    df = pd.read_csv('tmp_data' + stat_name + '.csv')
    df = normalize_proba(df)
    return df


def print_dist(best_dist):
    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
    dist_str = '{}({})'.format(best_dist[0].name, param_str)
    return dist_str


def sample_pdf(df, size=1, add_noise=True):  
    samples = np.random.choice(df['value'], size=size, replace=True, p=df['proba'])
    
    if add_noise:
        step = df['value'][1] - df['value'][0]
        noise = (np.random.random(len(samples)) - 0.5) * step
        samples = samples + noise

    return samples


def show_distribution(df, col_name):
    """normalize distribution and return dataframe 
        with probabilities and values of parameter
        from the input dataframe
    """
    
    d = df[col_name].to_list()
    hist = plt.hist(d, bins=100)
    plt.close()

    hist_data = {'proba': hist[0], 'value': hist[1][:100]}
    hist_df = pd.DataFrame(hist_data)
    hist_df_norm = normalize_proba(hist_df, min_value=None, max_value=None)

    return hist_df_norm


def sample_types_discrete(df, col='neighbour_part_types', size=1):
    all_types = list()
    lists = df[col]
    
    for lst in lists:
        all_types.extend(lst)
        
    cnt = Counter(all_types)
    probas = np.array(list(cnt.values()))
    probas = probas / probas.sum()
    samples = np.random.choice(list(cnt.keys()), size=size, replace=True, p=probas)
    return samples
