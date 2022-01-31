from random import random
from math import ceil

n = 10000000
x = 44.700935174964826
x_approx = 0

for i in range(n):
    r = random()
    x_approx += int(x)/n
    if r < x-int(x):
        x_approx += 1/n

print(x_approx)
