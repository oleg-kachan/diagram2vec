import numpy as np
from scipy.stats import skew as skewness, kurtosis

# dictionary of quantities computable out of an intervals of a persistence diagram
quantities = {
    "persistence": lambda x: x[:,1] - x[:,0],
    "midlife": lambda x: (x[:,0] + x[:,1]) / 2,
    "multlife": lambda x: x[:,1] / x[:,0]
}

# dictionary of statistics
stats = {
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "skewness": skewness,
    "kurtosis": kurtosis,
    "median": np.median,
    "entropy": lambda x: -(x @ log(x)),
    "count": lambda x: x.shape[0]
}

# dictionary of confidence interval functions
functions = {
    "linear": lambda x, **kwargs: linear(x, **kwargs),
    "pearson": lambda x, **kwargs: pearson(x, **kwargs)
}

# linear condidence function
def linear(x, a=0, b=0):
    return x * a + b

# Pearson correlation-dependent condidence function
def pearson(x, n=150, z_alpha=1.96):
    def z(x):
        return np.arctanh(x)

    def r(x):
        return np.tanh(x)      

    return (r(z(x) + z_alpha/np.sqrt(n - 3)) - r(z(x) - z_alpha/np.sqrt(n - 3))) / 2

# zero-robust logarithm function
def log(x):
    return np.log(x + 1e-100)

def _persistence(diagram, curve, quantity, thresholds):

    # TODO: return list of critical points if threshold is None
    if thresholds==None:
        critical_points = np.unique(diagram.reshape(-1))
    else:
        critical_points = thresholds

    x = {}
    for t in critical_points:
        x[t] = 0

    for t in critical_points:
        
        for interval in diagram:
            start, end = interval[0], interval[1]
            
            # persistence curves
            if curve=="persistence":

                if quantity=="betti":
                    x_inc = 1
                else:
                    x_inc = quantities[quantity](interval.reshape(1,-1)).reshape(-1)[0]
                
            # entropy curves
            elif curve=="entropy":

                p = quantities[quantity](interval.reshape(1,-1)).reshape(-1)[0]
                p_norm = quantities[quantity](diagram).sum()
                    
                p = p / p_norm
                x_inc = -(p * log(p))
            
            if ((start <= t) & (t < end)):
                x[t] = x[t] + x_inc
                
    keys, values = zip(*x.items())

    return np.array(values)


def _curve(diagram, m=101, k_max=None, curve_type="persistence", quantity="betti", f="linear", **kwargs):
    
    if (type(diagram) == list) & (type(diagram[0]) == np.ndarray):
        diagram = [diagram]

    if k_max is None:
        k_max = len(diagram[0]) - 1

    if type(m) == int:
        thresholds = list(np.linspace(0, 1, m))
    elif (type(m) == list) or (type(m) == np.ndarray):
        thresholds = list(m)
    
    # init output matrix
    curve = np.zeros((len(diagram), k_max+1, m))

    for i in range(len(diagram)):
        for k in range(k_max+1):

            # filter a diagram according to a function
            p = quantities["persistence"](diagram[i][k])
            diagram_ik = diagram[i][k][p > functions[f](p, **kwargs)]

            curve[i,k,:] = _persistence(diagram_ik[::-1], curve_type, quantity, thresholds)

    return curve


def statistics(diagram, k_max=None, quantity="persistence", statistics=["mean", "sum"], f="linear", **kwargs):
    """
    Return an i x k x n array of statistics of specified quantities of
    k-dimensional persistence diagram.

    Parameters
    ----------
    diagram: list of ndarrays
        list of arrays of k-dimensional persistence diagram
    k_max: int, optional
    quantity: {'persistence', 'midlife', or 'multlife'}, optional
    statistics: ['min', 'max', 'sum', 'mean', 'std', 'skewness', 'kurtosis', 'median', 'entropy', 'count'], optional
    f: {'linear'}, optional
        confidence interval function
    a: float, optional
    b: float, optional
    n: int, optional
    z_alpha: float, optional
    w_max: float, optional

    Returns
    -------
    out: ndarray
        i x k x n array of statistics of specified quantities of
        k-dimensional persistence diagram.
    """

    if (type(diagram) == list) & (type(diagram[0]) == np.ndarray):
        diagram = [diagram]

    if k_max is None:
        k_max = len(diagram[0]) - 1

    # init statistics matrix
    matrix_statistics = np.zeros((len(diagram), k_max+1, len(statistics)))
    
    for i in range(len(diagram)):
        for k in range(k_max+1):

            # filter a diagram according to a confidence function
            p = quantities["persistence"](diagram[i][k])
            diagram_ik = diagram[i][k][p > functions[f](p, **kwargs)]

            # compute a quantity out of a diagram
            q = quantities[quantity](diagram_ik)

            # compute the quantity's statistics
            for j, statistic in enumerate(statistics):
                matrix_statistics[i,k,j] = stats[statistic](q)

    # replace nans
    matrix_statistics = np.nan_to_num(matrix_statistics)

    return matrix_statistics


def euler_curve(diagram, m=101, k_max=None, f="linear", **kwargs):
    """
    Return an m-dimensional array of Euler characteristic up to dimension k_max.

    Parameters
    ----------
    diagram: list of ndarrays
        list of arrays of k-dimensional persistence diagram
    m: int or list of floats or ndarray of floats, optional
    k_max: int, optional
    f: {'linear'}, optional
        confidence interval function
    a: float, optional
    b: float, optional
    n: int, optional
    z_alpha: float, optional

    Return
    ------
    out: ndarray
        n array of statistics of specified quantities of
        k-dimensional persistence diagram.
    """

    return _curve(diagram, m, k_max, "persistence", "betti", f, **kwargs).sum(axis=1)


def persistence_curve(diagram, m=101, k_max=None, quantity="betti", f="linear", **kwargs):
    """
    Return an (k x m)-dimensional array of specified quantity up to dimension k=k_max.
    """

    return _curve(diagram, m, k_max, "persistence", quantity, f, **kwargs)


def entropy_curve(diagram, m=101, k_max=None, quantity="persistence", f="linear", **kwargs):
    """
    Return an (k x m)-dimensional array of a entropy of specified quantity up to dimension k=k_max.
    """

    return _curve(diagram, m, k_max, "entropy", quantity, f, **kwargs)