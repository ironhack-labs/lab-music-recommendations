"""Browser microphone recorder backed by the existing in-memory local RAG.

The web server listens on loopback only. No microphone is accessed by Python.
Chrome asks permission only when the user presses the HTML Record button.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading, json, uuid, urllib.parse, io

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:16px system-ui;margin:16px;color:#17354a}button{background:#d24b15;color:white;border:3px solid #ffcb55;border-radius:8px;padding:14px 24px;font-size:18px;cursor:pointer}button:disabled{opacity:.5}#status{padding:12px;background:#f1f5f8;margin:12px 0}audio{display:block;margin:8px 0;width:95%}.track{padding:10px;border-bottom:1px solid #ddd}
</style></head><body><h3>Phone speaker → Chrome microphone → local recommendations</h3>
<p>Play your phone speaker nearby. Click Record and choose <b>Allow</b> if Chrome asks. Pause the phone after the 20-second countdown ends.</p>
<p>Returns the 5 closest tracks from the local catalogue. Genre tags describe the reference songs; the search compares audio and does not identify the Spotify song.</p>
<button id="record">Record 20 seconds and recommend</button><div id="status">Ready. Nothing is recording.</div><div id="results"></div>
<script>
const button=document.getElementById('record'), status=document.getElementById('status'), results=document.getElementById('results');
fetch('/status').then(r=>r.json()).then(d=>{status.textContent=`Ready. ${d.tracks} reference tracks available. Nothing is recording.`;}).catch(()=>{status.textContent='Recorder server unavailable. Restart the local recorder.';});
function wavBlob(parts,rate){
 let n=parts.reduce((s,x)=>s+x.length,0),buffer=new ArrayBuffer(44+n*2),v=new DataView(buffer),offset=44;
 const text=(p,s)=>{for(let i=0;i<s.length;i++)v.setUint8(p+i,s.charCodeAt(i));};
 text(0,'RIFF');v.setUint32(4,36+n*2,true);text(8,'WAVE');text(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,'data');v.setUint32(40,n*2,true);
 for(const part of parts)for(const s of part){const x=Math.max(-1,Math.min(1,s));v.setInt16(offset,x<0?x*32768:x*32767,true);offset+=2;}
 return new Blob([buffer],{type:'audio/wav'});
}
function player(url,label){const p=document.createElement('p');p.textContent=label;results.appendChild(p);const a=document.createElement('audio');a.controls=true;a.src=url;results.appendChild(a);}
button.onclick=async()=>{
 let stream,context,source,processor,timer;
 button.disabled=true;results.replaceChildren();
 try{
  status.textContent='Waiting for Chrome microphone permission. Choose Allow.';
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Microphone API unavailable. Open this page on localhost in Chrome.');
  stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
  context=new AudioContext();await context.resume();source=context.createMediaStreamSource(stream);processor=context.createScriptProcessor(4096,1,1);
  let parts=[];processor.onaudioprocess=e=>parts.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(processor);processor.connect(context.destination);
  let remaining=20;status.textContent='Recording: 20 seconds remaining — keep the phone playing.';
  timer=setInterval(()=>{remaining--;status.textContent=remaining>0?`Recording: ${remaining} seconds remaining.`:'Recording finished — PAUSE YOUR PHONE. Finding recommendations…';},1000);
  await new Promise(resolve=>setTimeout(resolve,20000));
  clearInterval(timer);source.disconnect();processor.disconnect();stream.getTracks().forEach(t=>t.stop());
  const rate=context.sampleRate;await context.close();context=null;
  if(!parts.length)throw Error('The browser returned no audio samples. Check your microphone.');
  const response=await fetch('/record',{method:'POST',headers:{'Content-Type':'audio/wav'},body:wavBlob(parts,rate)});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Recording analysis failed.');
  player(data.recording_url,'Your captured recording — press ▶ to check it:');
  if(data.warning){const p=document.createElement('p');p.textContent=data.warning;results.appendChild(p);}
  for(const m of data.matches){const div=document.createElement('div');div.className='track';div.textContent=`${m.title||m.track_id} — ${m.artist||'Unknown artist'} | ${m.genre||m.instruments||'Unlabelled'} | similarity ${m.similarity.toFixed(3)}`;results.appendChild(div);player(m.audio_url,'▶ Play this recommended track:');}
  status.textContent=data.matches.length?'Done. Pause your phone and use the ▶ players below.':'Recorded, but no recommendations were generated; check the recording level.';
 }catch(e){status.textContent='Could not record: '+e.message+' Use Chrome microphone permission or upload an audio file instead.';}
 finally{clearInterval(timer);if(stream)stream.getTracks().forEach(t=>t.stop());if(context)await context.close();button.disabled=false;}
};
</script></body></html>'''

_server = globals().get('_server')

def start_browser_recorder(rag, project):
    global _server
    project=Path(project).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,code,payload,kind='application/json'):
            body=json.dumps(payload).encode() if kind=='application/json' else payload
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            route=urllib.parse.urlsplit(self.path).path
            if route=='/': return self.send(200,PAGE.encode(),'text/html; charset=utf-8')
            if route=='/status':return self.send(200,{'ready':True,'tracks':self.server.rag.collection.count()})
            if route.startswith('/audio/'):
                key=route[len('/audio/'):];p=self.server.audio_files.get(key)
                if p is None:return self.send(404,{'error':'Unknown audio'})
                kind='audio/wav' if p.suffix.lower()=='.wav' else 'audio/mpeg'
                return self.send(200,p.read_bytes(),kind)
            self.send(404,{'error':'Not found'})
        def do_POST(self):
            origin=self.headers.get('Origin')
            expected=f'http://127.0.0.1:{self.server.server_port}'
            if origin and origin!=expected:return self.send(403,{'error':'Use the recorder page itself.'})
            if self.path!='/record':return self.send(404,{'error':'Not found'})
            size=int(self.headers.get('Content-Length','0'))
            if not 44<size<8*1024*1024:return self.send(400,{'error':'Invalid recording size'})
            if self.headers.get_content_type()!='audio/wav':return self.send(400,{'error':'Expected WAV'})
            try:
                import soundfile as sf
                import numpy as np
                payload=self.rfile.read(size);samples,rate=sf.read(io.BytesIO(payload),dtype='float32')
                if samples.ndim!=1 or not np.isfinite(samples).all():raise ValueError('Invalid mono audio')
                duration=len(samples)/rate
                if not 18<=duration<=22:raise ValueError('Expected a roughly twenty-second recording; refresh the recorder page.')
                folder=project/'data/microphone_queries';folder.mkdir(exist_ok=True)
                key='query_'+uuid.uuid4().hex;path=folder/(key+'.wav');path.write_bytes(payload)
                rms=float(np.sqrt(np.mean(samples*samples)))
                self.server.audio_files[key]=path
                warning='Very quiet recording: move the phone closer and try again.' if rms<.002 else ''
                with self.server.query_lock:
                    matches=[] if warning else self.server.rag.recommend(path,k=5)
                public=[]
                for m in matches:
                    track_key='track_'+m['track_id'];self.server.audio_files[track_key]=Path(m['path'])
                    public.append({k:m.get(k,'') for k in ['track_id','title','artist','genre','bpm','instruments','similarity']})
                    public[-1]['audio_url']='/audio/'+track_key
                self.send(200,{'recording_url':'/audio/'+key,'rms':rms,'warning':warning,'matches':public})
            except Exception as exc:self.send(500,{'error':str(exc)})

    if _server is not None:
        _server.RequestHandlerClass=Handler
        _server.rag=rag
        return f'http://127.0.0.1:{_server.server_port}/'
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.rag=rag;server.audio_files={};server.query_lock=threading.Lock()
    threading.Thread(target=server.serve_forever,daemon=True).start()
    _server=server
    return f'http://127.0.0.1:{server.server_port}/'
