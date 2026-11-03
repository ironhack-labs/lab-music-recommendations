"""Download a bounded, genre-balanced FMA sample using HTTP ZIP ranges.

Original source: https://github.com/mdeff/fma . Audio licenses are retained
per track. No full 24 GB archive is downloaded.
"""
from pathlib import Path
import io, urllib.request, zipfile, json, csv, ast, random, sys
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')

ROOT = Path(__file__).resolve().parent

class RemoteZip(io.RawIOBase):
    def __init__(self, name):
        self.url = 'https://os.unil.cloud.switch.ch/fma/' + name
        self.pos = 0
        req = urllib.request.Request(self.url, headers={'Range': 'bytes=-65536'})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206: raise RuntimeError('Server must support byte ranges')
            self.size = int(r.headers['Content-Range'].split('/')[-1])
            self.tail = r.read()
        self.downloaded = len(self.tail)
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos
    def read(self, n=-1):
        n = min(n if n >= 0 else self.size-self.pos, self.size-self.pos)
        if n <= 0: return b''
        start = self.pos
        if start >= self.size-len(self.tail):
            data = self.tail[start-(self.size-len(self.tail)):][:n]
        else:
            if n > 50*1024*1024: raise RuntimeError('Refusing oversized range')
            # A query parameter avoids intermediary caches serving another range.
            req = urllib.request.Request(self.url+f'?range={start}-{start+n-1}',
                                         headers={'Range':f'bytes={start}-{start+n-1}'})
            with urllib.request.urlopen(req, timeout=90) as r:
                expected=f'bytes {start}-{start+n-1}/{self.size}'
                if r.status != 206 or r.headers.get('Content-Range') != expected:
                    raise RuntimeError('Incorrect HTTP range response')
                data=r.read(n+1)
            if len(data)!=n: raise RuntimeError('Truncated range')
            self.downloaded += len(data)
        self.pos += len(data)
        return data

def metadata():
    folder=ROOT/'data/fma'; folder.mkdir(parents=True, exist_ok=True)
    if not (folder/'tracks.csv').exists():
        remote=RemoteZip('fma_metadata.zip')
        with zipfile.ZipFile(remote) as z:
            for name in ('tracks.csv','genres.csv'):
                member=next(x for x in z.namelist() if x.endswith('/'+name))
                (folder/name).write_bytes(z.read(member))
        print('Metadata downloaded:',remote.downloaded,flush=True)
    with (folder/'genres.csv').open(encoding='utf-8') as f:
        genres={int(x['genre_id']):x['title'] for x in csv.DictReader(f)}
    with (folder/'tracks.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f); groups=next(reader); fields=next(reader); next(reader)
        keys=list(zip(groups,fields))
        rows=[dict(zip(keys,r)) for r in reader]
    return folder,genres,rows

def main():
    folder,genres,rows=metadata()
    if '--inspect' in sys.argv:
        print('Genres:',json.dumps(genres,ensure_ascii=False)); print('Keys:',list(rows[0])); return
    targets=['Jazz','Latin','Classical','Blues','Rock','Pop','Electronic','Hip-Hop','Folk','Soul-RnB']
    random.Random(42).shuffle(rows)
    remote=RemoteZip('fma_medium.zip')
    result=[]; used=set(); failures=[]
    if '--latin-only' in sys.argv:
        targets=['Latin']
        result=[r for r in json.loads((ROOT/'catalogue_fma.json').read_text()) if r['genre']!='Latin']
        used={r['track_id'].removeprefix('fma_') for r in result}
    with zipfile.ZipFile(remote) as z:
        members={Path(x).stem:x for x in z.namelist() if x.endswith('.mp3')}
        for genre in targets:
            count=0; artists={}
            for r in rows:
                tid=r.get(('','')) or r.get(('track','id'))
                # First column is the track id and has an empty two-level header.
                tid=next(iter(r.values()))
                key=f'{int(tid):06d}'
                labels=[genres.get(i,str(i)) for i in ast.literal_eval(r[('track','genres_all')] or '[]')]
                artist=r[('artist','name')]
                license=r[('track','license')]
                matches_genre=genre in labels
                if genre=='Latin':
                    matches_genre=bool(set(labels)&{'Latin','Latin America','Brazilian','Cumbia','Salsa','Tango'})
                if not matches_genre or key not in members or key in used: continue
                if artists.get(artist,0)>=(3 if genre=='Latin' else 1): continue
                if not any(x in license for x in ('Attribution','Public Domain','CC0')): continue
                path=folder/'audio'/(key+'.mp3'); path.parent.mkdir(exist_ok=True)
                try:
                    if not path.exists():
                        data=z.read(members[key])
                        if len(data)>3*1024*1024: continue
                        path.write_bytes(data)
                    result.append({'track_id':'fma_'+key,'path':path.relative_to(ROOT).as_posix(),
                        'title':r[('track','title')], 'artist':artist, 'genre':genre,
                        'genres':', '.join(labels),'license':license,
                        'source':'FMA medium / https://github.com/mdeff/fma',
                        'source_url':'https://github.com/mdeff/fma', 'split':'reference'})
                    used.add(key);artists[artist]=artists.get(artist,0)+1;count+=1
                    (ROOT/'catalogue_fma.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                    print(genre,count,key,r[('track','title')],flush=True)
                    if count>=20: break
                except Exception as e:
                    failures.append({'track':key,'error':str(e)}); print('Skipped',key,str(e),flush=True)
            print('Genre complete',genre,count,flush=True)
    (folder/'download_manifest.json').write_text(json.dumps({'source':remote.url,
        'downloaded_bytes':remote.downloaded,'tracks':len(result),'failures':failures},indent=2))
    print('COMPLETE',len(result),'tracks; downloaded',remote.downloaded,flush=True)

if __name__=='__main__': main()
