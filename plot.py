import re

file = open('benchmark.txt', 'r')

dc = {}
current = None
for line in file:
    if line[0] == '#':
        dc[current].append(float(line[2:]))
    elif (match := re.search('\s(\d+).*\s(\d+).*\s(\d+).*\s(\d+)', line)):
        n, nprime, k1, k2 = (match.group(i) for i in range(1,5))
        current = (n, nprime, k1, k2)
        dc[current] = []

print(dc)
