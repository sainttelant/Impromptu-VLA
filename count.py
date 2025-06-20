import json
with open('nuscenes_test.json', 'r') as f:
    data = json.load(f)

count = {}

for d in data:
    id = d['id']
    if id not in count:
        count[id] = 0
    count[id] += 1

num = 0
cc = {}
for id, c in count.items():
    if c not in cc:
        cc[c] = 0
    cc[c] += 1
print(cc)
