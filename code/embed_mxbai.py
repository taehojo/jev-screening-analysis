# ASReview dory의 mxbai 파이프라인(elas_h3의 특징 추출기)으로 리뷰별 임베딩을 한 번 계산해 저장
import os, sys, json, time
import numpy as np, pandas as pd
from asreviewcontrib.dory.feature_extractors.sentence_transformer_embeddings import MXBAI
from sentence_transformers import SentenceTransformer
import torch; torch.set_num_threads(int(os.environ.get('EMB_THREADS', '16')))
recf = sys.argv[1]
R = json.load(open(recf)); by = {}
for r in R: by.setdefault(r['review'], []).append(r)
for k in sorted(by, key=lambda k: len(by[k])):
    out = f'emb_mxbai/{k}.npy'
    if os.path.exists(out): continue
    v = by[k]; df = pd.DataFrame({'title': [r['title'] for r in v], 'abstract': [r['abstract'] for r in v]})
    # dory MXBAI(normalize=True)와 같은 단계: 제목+' '+초록 → mxbai 임베딩 → L2 정규화.
    # 단, 새 transformers가 체크포인트를 float16으로 올려 CPU에서 수십 배 느려지므로 float32로 강제한다(ASReview 원래 환경과 같은 정밀도).
    pipe = MXBAI(normalize=True, verbose=False)
    pipe.named_steps['sentence_transformer'].__dict__['_model'] = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu', model_kwargs={'torch_dtype': torch.float32})
    t0 = time.time(); E = pipe.fit_transform(df)
    np.save(out, np.asarray(E, dtype=np.float32)); print(k, len(v), f'{time.time()-t0:.0f}s', flush=True)
print('done', flush=True)
