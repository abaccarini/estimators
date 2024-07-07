import math

x_i = [2.0, 4.0, 5.0, 6.0, 8.0]


def mean(x_i):
    return sum(x_i)/len(x_i)
def variance(x_i):
    mu = sum(x_i)/len(x_i)
    accum = 0
    for xi in x_i:
        accum += (xi - mu)*(xi - mu)
    return accum / len(x_i)

print(x_i)
print(mean(x_i),variance(x_i))
x_ii =[x * 0.9 for x in x_i] 
print(x_ii)
print(mean(x_ii),variance(x_ii))