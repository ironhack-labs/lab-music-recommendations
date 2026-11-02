# Music catalog

Eight distinct music recordings from the [librosa example collection](https://librosa.org/doc/main/recordings.html).
The lab allows choosing another online dataset; this small collection keeps setup
practical for a four-hour lab. No pitch-shifted copies or environmental sounds are
used as music candidates. `miaow_16k.wav` belongs only to the original tutorial.

Run `python scripts/download_music.py` from the project root. Downloads are checked
against SHA-256 hashes from librosa's example registry. Original credit and license
notices are saved under `attributions/`. Each recording retains its own license;
read its notice before reuse. These notices are also linked in `catalog.json`.

`catalog.json` records title, artist, source URLs, checksums, and relative audio
paths. Genre/mood/instrument tags are manually curated for this exercise, not
ground-truth annotations or automatic model predictions. They support a controlled
preference vocabulary; arbitrary language and negation are not understood.

Only the first 30 seconds of each recording are embedded, in mono at 22050 Hz.
The original downloaded files are unchanged. The generated `music_index.npz`
contains standardized, normalized audio vectors and catalog metadata. It is
rebuilt when the catalog, audio contents, or feature version changes.
