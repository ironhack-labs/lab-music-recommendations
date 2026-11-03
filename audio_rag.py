"""Local music retrieval: PANNs Cnn14 + Chroma, no external inference API."""
from pathlib import Path
import os, sys, json, hashlib, math, zipfile
ROOT = Path(__file__).resolve().parent
for name in ('mpl_cache', 'numba_cache'):
    (ROOT/name).mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT/'mpl_cache')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'numba_cache')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
for p in (ROOT/'vendor_py', ROOT.parent/'Local_Image_RAG'/'vendor_py'):
    if str(p) not in sys.path: sys.path.append(str(p))
import numpy as np
from scipy.signal import resample_poly

SR = 32000
WINDOW_SECONDS = 10
MODEL_NAME = 'panns-cnn14-audioset-0.431'
CHECKPOINT = ROOT/'models/Cnn14_mAP=0.431.pth'

def load_catalogue():
    rows=json.loads((ROOT/'catalogue.json').read_text())
    for row in rows: row['path']=str((ROOT/row['path']).resolve())
    return rows

def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_audio(path):
    import soundfile as sf
    wave, source_sr = sf.read(str(path),dtype='float32',always_2d=True)
    if not wave.size or not np.isfinite(wave).all(): raise ValueError('Empty/nonfinite audio')
    mono = wave.mean(axis=1)
    if source_sr != SR:
        g = math.gcd(source_sr,SR)
        mono = resample_poly(mono,SR//g,source_sr//g).astype(np.float32)
    peak = np.max(np.abs(mono))
    if peak > 1: mono = mono / peak
    return np.ascontiguousarray(mono,dtype=np.float32)

def windows(wave):
    """At most 3 evenly spaced ten-second windows; short audio is zero padded."""
    n = WINDOW_SECONDS * SR
    starts = np.unique(np.linspace(0,max(0,len(wave)-n),min(3,max(1,math.ceil(len(wave)/n))),dtype=int))
    clips=[]
    for start in starts:
        clip=wave[start:start+n]
        clips.append(np.pad(clip,(0,max(0,n-len(clip)))))
    return np.asarray(clips,dtype=np.float32)

class AudioEncoder:
    def __init__(self):
        import torch
        import importlib, importlib.util, types
        # Import the model definitions without the legacy package __init__,
        # which otherwise writes into the user's home and invokes wget.
        spec=importlib.util.find_spec('panns_inference')
        if spec is None: raise ImportError('Install panns-inference first')
        namespace='_local_audio_panns'
        if namespace not in sys.modules:
            package=types.ModuleType(namespace)
            package.__path__=[str(Path(spec.origin).parent)]
            sys.modules[namespace]=package
        Cnn14=importlib.import_module(namespace+'.models').Cnn14
        if not CHECKPOINT.is_file(): raise FileNotFoundError('Model weights missing: '+str(CHECKPOINT))
        # Instantiate directly: the older AudioTagging wrapper invokes wget and
        # torch.load defaults that are unsuitable for this Windows setup.
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        self.torch=torch
        self.model=Cnn14(sample_rate=SR,window_size=1024,hop_size=320,
                         mel_bins=64,fmin=50,fmax=14000,classes_num=527)
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.model.eval()

    def embed(self,wave):
        vectors=[]
        # One window per inference bounds CPU memory.
        for clip in windows(wave):
            with self.torch.inference_mode():
                output=self.model(self.torch.from_numpy(clip[None,:]),None)
            vectors.append(output['embedding'][0].cpu().numpy())
        vector=np.mean(vectors,axis=0).astype(np.float32)
        norm=np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm<=0: raise ValueError('Invalid embedding')
        vector/=norm
        assert vector.shape==(2048,)
        return vector.tolist()

class MusicRAG:
    def __init__(self,encoder,collection_name=None):
        import chromadb
        self.encoder=encoder
        self.client=chromadb.PersistentClient(path=str(ROOT/'chroma_audio'))
        if collection_name is None:
            config=ROOT/'active_catalogue.json'
            collection_name=json.loads(config.read_text())['collection'] if config.exists() else 'tiny_aam_cnn14_v1'
        self.collection=self.client.get_or_create_collection(collection_name,
            embedding_function=None,metadata={'hnsw:space':'cosine','model':MODEL_NAME,'sr':SR})

    def index(self,catalogue):
        old=set(self.collection.get()['ids']);seen=set()
        for row in catalogue:
            digest=file_hash(row['path'])
            if digest in seen:continue
            seen.add(digest)
            metadata={k:v for k,v in row.items() if isinstance(v,(str,int,float,bool))}
            metadata.update(sha256=digest,model=MODEL_NAME)
            if digest in old:self.collection.update(ids=[digest],metadatas=[metadata])
            else:self.collection.upsert(ids=[digest],embeddings=[self.encoder.embed(load_audio(row['path']))],metadatas=[metadata])
            print('Indexed:',row['track_id'],flush=True)
        # Do not delete unseen records automatically. A different catalogue
        # should use a new named collection, preserving existing work.
        return self.collection.count()

    def recommend(self,path,k=5,min_bpm=None,max_bpm=None,instrument=None):
        if k<1: raise ValueError('k must be positive')
        if min_bpm is not None and max_bpm is not None and min_bpm>max_bpm:
            raise ValueError('Minimum BPM must not exceed maximum BPM')
        if self.collection.count()==0:raise ValueError('Index the catalogue first')
        digest=file_hash(path)
        result=self.collection.query(query_embeddings=[self.encoder.embed(load_audio(path))],
            n_results=self.collection.count(),include=['metadatas','distances'])
        hits=[];seen_tracks=set()
        for meta,d in zip(result['metadatas'][0],result['distances'][0]):
            if meta['sha256']==digest or meta['track_id'] in seen_tracks:continue
            if min_bpm is not None and (meta.get('bpm',-1)<min_bpm):continue
            if max_bpm is not None and (meta.get('bpm',float('inf'))>max_bpm):continue
            if instrument and instrument.casefold() not in meta.get('instruments','').casefold():continue
            seen_tracks.add(meta['track_id']);hits.append(dict(meta,similarity=float(1-d)))
            if len(hits)>=k:break
        return hits

def explain(matches):
    """Grounded, deterministic explanation: never invent an LLM listening claim."""
    if not matches:return 'No tracks meet these filters. Broaden your preferences.'
    return '\n'.join(f"{i+1}. {m['track_id']}: acoustic cosine similarity {m['similarity']:.3f}; "
                     f"tempo {m.get('bpm','unknown')} BPM; instruments: {m.get('instruments','unknown')}."
                     for i,m in enumerate(matches))
