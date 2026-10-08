// 0단계 판정: 묶음 크기별 AUC와 단일 요청 대비 순위 상관
import fs from 'node:fs';
const single = new Map(JSON.parse(fs.readFileSync('../screen/screen_jev.json', 'utf8')).filter((d) => d.ok).map((d) => [d.pmid, d]));
function auc(pos, neg) { let s = 0; for (const p of pos) for (const n of neg) s += p > n ? 1 : p === n ? 0.5 : 0; return s / (pos.length * neg.length); }
const rank = (x) => { const idx = x.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = []; let i = 0; while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
const sp = (a, b) => { const ra = rank(a), rb = rank(b), ma = ra.reduce((x, y) => x + y) / a.length, mb = rb.reduce((x, y) => x + y) / b.length; let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; } return n / Math.sqrt(da * db); };
const workAt = (D, r) => { const s = [...D].sort((a, b) => b.p - a.p); const nPos = D.filter((d) => d.label === 1).length; let c = 0; for (let i = 0; i < s.length; i++) { if (s[i].label === 1) c++; if (c >= Math.ceil(r * nPos)) return (i + 1) / s.length; } return 1; };
const S = [...single.values()];
console.log(`단일(batch=1) n=${S.length}: AUC ${auc(S.filter((d) => d.label).map((d) => d.p), S.filter((d) => !d.label).map((d) => d.p)).toFixed(3)}  재현율95% 읽기 ${(100 * workAt(S, 0.95)).toFixed(1)}%`);
for (const f of process.argv.slice(2)) {
  const D = JSON.parse(fs.readFileSync(f, 'utf8')).filter((d) => d.ok && single.has(d.id));
  const pos = D.filter((d) => d.label === 1).map((d) => d.p), neg = D.filter((d) => d.label === 0).map((d) => d.p);
  const S2 = D.map((d) => single.get(d.id));
  console.log(`${f} n=${D.length}: AUC ${auc(pos, neg).toFixed(3)}  (같은 기록 단일 AUC ${auc(S2.filter((d) => d.label).map((d) => d.p), S2.filter((d) => !d.label).map((d) => d.p)).toFixed(3)})  순위 상관 ${sp(D.map((d) => d.p), S2.map((d) => d.p)).toFixed(3)}  재현율95% 읽기 ${(100 * workAt(D, 0.95)).toFixed(1)}%  평균 토큰 ${Math.round(D.reduce((s, d) => s + (d.tokens || 0), 0) / D.length)}`);
}
