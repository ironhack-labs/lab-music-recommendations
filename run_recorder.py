"""Open the saved local music index in the browser recorder."""
from pathlib import Path
import threading
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_rag import AudioEncoder, MusicRAG
from browser_microphone import start_browser_recorder

if __name__ == '__main__':
    project = Path(__file__).resolve().parent
    print('Loading saved audio model and index...', flush=True)
    rag = MusicRAG(AudioEncoder())
    if not rag.collection.count():
        raise RuntimeError('Saved music index is empty; run the notebook indexing step.')
    url = start_browser_recorder(rag, project)
    (project/'jupyter_runtime'/'recorder_url.txt').write_text(url, encoding='utf-8')
    print('READY: '+url, flush=True)
    threading.Event().wait()
