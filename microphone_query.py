"""Explicit, user-triggered microphone recording for local audio queries."""
from pathlib import Path
import datetime, json
import numpy as np

def input_devices():
    import sounddevice as sd
    return [(i,d['name']) for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]

def capture_query(folder,device=None,seconds=10):
    """Call only after the user presses Record. Never records during import."""
    import sounddevice as sd
    import soundfile as sf
    if not 1<=seconds<=30:raise ValueError('Record between 1 and 30 seconds.')
    info=sd.query_devices(device,kind='input')
    rate=int(info['default_samplerate'])
    sd.check_input_settings(device=device,channels=1,dtype='float32',samplerate=rate)
    print(f'Recording {seconds} seconds now. Keep the phone speaker near the laptop microphone.',flush=True)
    audio=sd.rec(int(seconds*rate),samplerate=rate,channels=1,dtype='float32',device=device)
    status=sd.wait()
    if status:raise RuntimeError('Audio capture had dropped samples: '+str(status))
    if not audio.size or not np.isfinite(audio).all():raise ValueError('Invalid microphone recording.')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=folder/('microphone_'+stamp+'.wav')
    sf.write(path,audio,rate,subtype='PCM_16')
    summary={'path':str(path),'device':info['name'],'sample_rate_hz':rate,
             'duration_seconds':seconds,'peak':float(np.max(np.abs(audio))),
             'rms':float(np.sqrt(np.mean(audio**2)))}
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return path,summary
