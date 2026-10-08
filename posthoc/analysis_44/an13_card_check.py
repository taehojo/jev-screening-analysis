# npj AN-0001-13: environment check. Runs the usage example of the model card (model_card_README.md) and compares the logits
# with the output printed in the card (6.9363, -8.2063, -8.7692, -12.3450, -10.4416, -15.8475). Public example text only.
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ['HF_HOME'] = os.path.join(HERE, 'hf_home'); os.environ['HF_HUB_OFFLINE'] = '1'
import torch, transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
card = open(os.path.join(HERE, 'model_card_README.md')).read()
query = re.search(r'query = "([^"]+)"', card).group(1)
articles = re.findall(r'^\t"(.+)",?$', card, flags=re.M)
path = json.load(open(os.path.join(HERE, 'model_download.json')))['local_path']
tok = AutoTokenizer.from_pretrained(path); model = AutoModelForSequenceClassification.from_pretrained(path).eval()
with torch.no_grad():
    enc = tok([[query, a] for a in articles], truncation=True, padding=True, return_tensors='pt', max_length=512)
    lg = model(**enc).logits.squeeze(dim=1).tolist()
exp = [6.9363, -8.2063, -8.7692, -12.3450, -10.4416, -15.8475]
res = {'transformers': transformers.__version__, 'torch': torch.__version__, 'n_articles': len(articles), 'logits': lg, 'card_logits': exp,
       'max_abs_diff': max(abs(a - b) for a, b in zip(lg, exp)), 'match_at_4dp': all(round(a, 4) == b for a, b in zip(lg, exp))}
json.dump(res, open(os.path.join(HERE, 'card_check.json'), 'w'), indent=1); print(res)
