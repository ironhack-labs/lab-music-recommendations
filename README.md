![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Music recommendations

The completed solution is in `lab-music-recommendations.ipynb`. It recommends from
eight real music recordings using written preferences, favorite track IDs, or an
audio query. Audio embeddings and TF-IDF metadata retrieval run locally; answers
are generated from retrieved metadata with a deterministic template. No API keys
or LLM downloads are needed.

## Run the completed solution

Use Python 3.10 or later from this directory:

```powershell
python -m pip install -r requirements.txt
python scripts/download_music.py
python music_recommendations.py build
python music_recommendations.py query --preferences "relaxed jazz piano" --top-k 3
python music_recommendations.py query --preferences "jazz piano" --seed trumpet
python music_recommendations.py query --audio data/music/choice.ogg --top-k 3
```

The music files and original attribution notices are included. The downloader
verifies their hashes, or downloads missing files. After setup, index building
and querying work offline. To add music, add a recording and its metadata to
`data/catalog.json`, then rebuild. Paths in the catalog are relative to `data/`.
See [catalog documentation](data/README.md) for dataset sources and limitations.

Open the notebook in Jupyter or VS Code, select the installed Python environment,
and run all cells. It demonstrates text retrieval, hybrid recommendations,
playback, a held-out audio query, and five explicit preference smoke checks.
The original Pinecone tutorial is retained as an optional, disabled section;
live Pinecone execution has not been verified. Cloud index deletion also has a
separate disabled flag.

```powershell
python -m unittest discover -s tests -v
python scripts/execute_notebook.py
```

Preferences use catalog words, such as piano, guitar, jazz, classical, or vocals.
Use `--genre classical` for an exact filter; repeat `--seed` for several favorite
tracks. Unknown preferences produce an error, and query/seed tracks are excluded.
This small catalog and its curated tags demonstrate the pipeline; the included
smoke checks are not an independent measure of subjective recommendation quality.

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
