from cmath import sqrt
import math
from scipy.special import lambertw
from cmath import exp
from scipy.stats import lognorm, norm
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

def lognormal_cf(t, mu, sigma):
    gamma = t * (sigma ** 2) * exp(mu) * 1j
    # print(lambertw(gamma)
    num = exp( (-1.0) * (pow(lambertw(gamma), 2) + 2 * lambertw(gamma)) / (2.0 * sigma ** 2) )
    den = sqrt(1 + lambertw(gamma))
    return num / den


sigma = 0.25
mu = 0
# x = np.linspace(lognorm.ppf(0.0001, sigma), lognorm.ppf(0.9999, sigma), 100)
x = np.linspace(0.01, 10.0,10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))


print("f_of_x =", lognorm.pdf(math.exp(mu - sigma*sigma), s=sigma, scale=math.exp(mu)))

# print(np.argmax(x))
# ax.plot(x, lognorm.pdf(x, s=sigma, scale=math.exp(mu)), 'r-', lw=5, alpha=0.6, label='lognorm pdf')
ax.plot(x, pdf, 'r-', lw=5, alpha=0.6, label='lognorm pdf')

space = [(xx, lognorm.pdf(xx, s=sigma, scale=math.exp(mu))) for xx in x]
# print(space) 
print("f_of_x =", lognorm.pdf(1.0, s=sigma, scale=math.exp(mu)))


math.sigma = sqrt(5)

print("norm f_of_x =", norm.pdf(0.0, loc=0, scale=sigma))




# rv = lognorm(sigma)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
# plt.show()

plt.savefig('test.pdf')

# for i in range(0, 5):
#     print(i, lognormal_cf(i, mu, sigma))

