import math
import matplotlib.pyplot as plt
"""
T: time | step rate
X: spread for time versus sine wave cycle
Y: amplitude of sine wave at given point X
C: complete cycles over X range
"""
C = 3
T = range(1000)
X = [C * (2 * math.pi * t) / len(T) for t in T]
Y = [math.sin(x) for x in X]

# Output and Label Graph
plt.plot(X,Y)
plt.xlim(0, max(X))
plt.ylim(1.1 * min(Y), 1.1 * max(Y))
plt.suptitle('Sine Wave Signal\n(%d cycles)' % C)
plt.show()
