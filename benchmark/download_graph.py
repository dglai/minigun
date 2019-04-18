import urllib.request
import os
import sys
import pickle as pkl

if not os.path.exists('data'):
    os.mkdir('data')
dataset = sys.argv[1]

url = 'https://github.com/tkipf/gcn/raw/master/gcn/data/ind.{}.graph'.format(dataset)
urllib.request.urlretrieve(url, '{}.graph'.format(dataset))  
with open('{}.graph'.format(dataset), 'rb') as f:
    g = pkl.load(f)

f = open('data/{}.txt'.format(dataset), 'w')
n, m = 0, 0
for k, v in g.items():
    if k > n: n = k
    for node in v:
        m += 1
        if node > n: n = node 

n += 1
print(n, m, file=f)
for k, v in g.items():
    for node in v:
        print(k, node, file=f)

f.close()
