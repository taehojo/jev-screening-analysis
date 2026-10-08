"""AN-0003-01 (analysis 51): extract the text of the obtained PDF with PyMuPDF (no network).
Writes sources/kusa_pdf_text.txt (pages separated by form-feed markers) and extract_meta.json."""
import json, hashlib, pymupdf
D = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0003/revision/analysis/AN-0003-01'
pdf = f'{D}/sources/ucl_eprint_10164959_kusa_main.pdf'
raw = open(pdf, 'rb').read()
doc = pymupdf.open(pdf)
pages = [p.get_text() for p in doc]
with open(f'{D}/sources/kusa_pdf_text.txt', 'w') as f:
    for i, t in enumerate(pages, 1):
        f.write(f'\n\f===== PAGE {i} =====\n{t}')
meta = {'pdf': pdf, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
        'n_pages': doc.page_count, 'metadata': doc.metadata,
        'pymupdf_version': pymupdf.VersionBind, 'chars_per_page': [len(t) for t in pages]}
json.dump(meta, open(f'{D}/extract_meta.json', 'w'), indent=1)
print(json.dumps(meta, indent=1))
