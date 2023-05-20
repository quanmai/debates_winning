import json
from utils.config import config
with open(config.original_f, 'r') as f:
# with open(config.filtered_f, 'r') as f:
    debates = json.load(f)
f.close()

# for k in debates.keys():
#     print(k)
with open('sample.json', 'w') as rf:
    # json.dump(debates['The-WWE-is-fake/1/'], rf)
    json.dump(debates['abortion-legal-or-illegal/1/'], rf)
rf.close()