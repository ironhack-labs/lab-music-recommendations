![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Music recommendations

## Completed local audio RAG solution

Open **lab-music-recommendations.ipynb**. The solution uses PANNs Cnn14 audio
embeddings, persistent local ChromaDB, cosine retrieval, exact-file deduplication
and self-match exclusion. It runs locally without Colab, API keys or Pinecone.
The browser recorder captures 20 seconds on a user click and returns five
playable recommendations with titles, artists and reference genre tags.

### Run on a fresh machine

Use Python 3.11 or newer with a PyTorch-supported platform. Tested locally with
Python 3.14 on Windows/CPU; the complete pipeline needs several GB of free
disk/RAM and initial downloads. Install requirements in a virtual environment.

```bash
python -m pip install -r requirements.txt
python setup_assets.py
python expand_catalogue.py
python build_expanded_index.py
```

The last command builds the separate 200-track FMA index and prints the local
recorder URL. Keep it running to use that recorder. Later, start it without
re-indexing with `python run_recorder.py`. In another terminal, run:

```bash
python -m notebook lab-music-recommendations.ipynb
```

Run cells top to bottom. Step 7B opens the 20-second browser recorder using the
active FMA collection. The preceding teaching examples and held-out evaluation
use a separate Tiny AAM collection (16 references, four held-out queries).
Chrome asks for microphone permission only after the user clicks Record.
Pause the phone after the countdown and use the returned audio players.
The optional Spotify embed is playback-only; Spotify audio is not downloaded.

### Catalogue and evidence

- FMA: 200 unique 30-second excerpts from 192 artists, 20 in each of ten groups:
  jazz, Latin, classical, blues, rock, pop, electronic, hip-hop, folk, soul/R&B.
  `catalogue_fma.json` preserves title, artist, dataset tags and per-track license.
  Downloads use checked byte ranges, not the full 24 GB FMA medium archive.
- `expanded_catalogue_validation.json`: four saved recording checks produced
  three distinct top-ranked tracks, versus one in the earlier 16-track index.
  Mean pairwise top-five Jaccard overlap fell from 0.603 to 0.185. This measures
  result diversity, not correctness or listener satisfaction.
- `verification.json` and `evaluation.csv`: original Tiny AAM evaluation.
  Instrument-overlap scores improved over a seeded random baseline, while
  tempo similarity did not. Human listening and larger held-out genre
  evaluation remain necessary.
- PANNs is a general audio encoder. Genre tags describe reference songs; this
  system does not identify the source Spotify song or guarantee its genre.
  Explanations use retrieved metadata, without a generative LLM.

Audio, model weights, Chroma storage, private recordings, credentials and
caches are excluded from Git. Notebook outputs are cleared for portability.
See [LOCAL_AUDIO_RAG.md](LOCAL_AUDIO_RAG.md) for implementation notes and sources.

### Sources and licenses

- [FMA](https://github.com/mdeff/fma): metadata CC BY 4.0; audio licenses vary
  and are retained per track. Some tracks are noncommercial/NoDerivatives.
- [Tiny AAM](https://doi.org/10.5281/zenodo.6771120): CC BY 4.0.
- [PANNs](https://github.com/qiuqiangkong/panns_inference) and
  [official checkpoint](https://zenodo.org/records/3987831).

---

## Getting Started

Follow the instructions provided in the notebook.

Read the instructions for each cell and provide your answers. Make sure to test your answers in each cell and save. Jupyter Notebook should automatically save your work progress. But it's a good idea to periodically save your work manually just in case.

## Deliverables

- Downloaded notebook with your responses to each of the exercises.


## Submission

- Upon completion, add your deliverables to git. 
- Then commit git and push your branch to the remote.
- Make a pull request and paste the PR link in the submission field in the Student Portal.

<br>

**Good luck!**
