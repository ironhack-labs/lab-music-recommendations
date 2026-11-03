"""Reproduce the verified official downloads and 20-track catalogue.

Run once before the notebook on a fresh machine. Does not install packages.
"""
from pathlib import Path
import hashlib, json, re, urllib.request, zipfile, wave

ROOT=Path(__file__).resolve().parent

def fetch(record,key,target):
    with urllib.request.urlopen('https://zenodo.org/api/records/'+record,timeout=30) as r:
        meta=json.load(r)
    entry=next(x for x in meta['files'] if x['key']==key)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or target.stat().st_size!=entry['size']:
        temp=target.with_suffix(target.suffix+'.part')
        with urllib.request.urlopen(entry['links']['self'],timeout=90) as source,temp.open('wb') as out:
            for chunk in iter(lambda:source.read(1024*1024),b''):out.write(chunk)
        temp.replace(target)
    h=hashlib.md5()
    with target.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if 'md5:'+h.hexdigest()!=entry['checksum']:raise ValueError('Checksum mismatch: '+key)
    target.with_suffix(target.suffix+'.source.json').write_text(json.dumps({
        'record':record,'file':key,'checksum':entry['checksum'],
        'url':entry['links']['self'],'metadata':meta['metadata']},indent=2))
    print('Verified:',key)

def main():
    import numpy as np
    fetch('3987831','Cnn14_mAP=0.431.pth',ROOT/'models/Cnn14_mAP=0.431.pth')
    archive=ROOT/'downloads/tinyAAM.zip'
    fetch('6771120','tinyAAM.zip',archive)
    target=ROOT/'data/tiny_aam';rows=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        mixes=sorted(x for x in names if x.startswith('audio-mixes-mp3/') and x.endswith('.mp3'))
        for i,name in enumerate(mixes):
            p=target/name
            if not p.resolve().is_relative_to(target.resolve()):raise ValueError('Invalid member')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
            tid=Path(name).stem.split('_')[0]
            beatname=f'annotations/{tid}_beatinfo.arff'
            beats=z.read(beatname).decode()
            (target/'annotations').mkdir(exist_ok=True)
            (target/beatname).write_text(beats)
            times=[float(line.split(',')[0]) for line in beats.splitlines() if re.match(r'^\d',line)]
            intervals=np.diff(times);bpm=round(60/float(np.median(intervals[intervals>0])),1)
            instruments=sorted({Path(x).stem.split('_',1)[1] for x in names
                if x.startswith(f'audio-multitracks-mp3/{tid}_') and x.endswith('.mp3')})
            rows.append({'track_id':tid,'path':p.relative_to(ROOT).as_posix(),
                'bpm':bpm,'instruments':', '.join(instruments),
                'split':'heldout' if i%5==0 else 'reference',
                'source':'Tiny AAM, Zenodo 6771120','license':'CC-BY-4.0'})
    (ROOT/'catalogue.json').write_text(json.dumps(rows,indent=2))
    # Small query file from the original lab, used for Steps 1 and 2 only.
    sample=ROOT/'data/miaow_16k.wav'
    if not sample.exists():
        with urllib.request.urlopen('https://raw.githubusercontent.com/ironhack-labs/lab-music-recommendations/main/data/miaow_16k.wav',timeout=30) as r:
            sample.write_bytes(r.read())
    print('Catalogue ready:',len(rows),'tracks. Open the notebook locally.')

if __name__=='__main__':main()
