
import scipy.spatial as ss
from scipy.special import digamma
from math import log, exp

INV_SQRT_2PI = 0.3989422804014327

def normal_pdf(x, mu, sigma):
    a = (x - mu) / sigma
    return INV_SQRT_2PI / sigma * exp(-(0.5) * a * a)

def lognormal_pdf(x, mu, sigma):
    a = (log(x) - mu) / sigma
    return INV_SQRT_2PI / (x * sigma) * exp(-(0.5) * a * a)
