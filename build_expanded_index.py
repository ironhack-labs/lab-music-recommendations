"""Build a separate FMA collection, evaluate saved queries, then serve it."""
from pathlib import Path
import sys, json, threading, collections
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from audio_rag import AudioEncoder, MusicRAG
from browser_microphone import start_browser_recorder

print('Loading audio model...',flush=True)
rag=MusicRAG(AudioEncoder(),collection_name='fma_balanced_cnn14_v1')
rows=json.loads((ROOT/'catalogue_fma.json').read_text(encoding='utf-8'))
for r in rows:r['path']=str(ROOT/r['path'])
assert len(rows)>=150, 'Do not activate an incomplete download'
print('Indexing',len(rows),'tracks',flush=True)
rag.index(rows)
assert rag.collection.count()==len(rows)
report={'count':len(rows),'genres':dict(collections.Counter(r['genre'] for r in rows)), 'queries':[]}
diagnostics=ROOT/'recording_diagnostics.json'
old=json.loads(diagnostics.read_text()) if diagnostics.exists() else []
for q in old:
    matches=rag.recommend(ROOT/'data/microphone_queries'/q['file'],k=5)
    report['queries'].append({'file':q['file'],'results':[{k:m.get(k,'') for k in
        ('track_id','title','artist','genre','similarity')} for m in matches]})
    print('QUERY',q['file'],[(m['track_id'],m.get('genre')) for m in matches],flush=True)
(ROOT/'expanded_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(ROOT/'active_catalogue.json').write_text(json.dumps({'collection':'fma_balanced_cnn14_v1',
    'catalogue':'catalogue_fma.json','tracks':len(rows),'genres':report['genres']},indent=2))
url=start_browser_recorder(rag,ROOT)
(ROOT/'jupyter_runtime/recorder_url.txt').write_text(url)
print('READY:',url,flush=True)
threading.Event().wait()
