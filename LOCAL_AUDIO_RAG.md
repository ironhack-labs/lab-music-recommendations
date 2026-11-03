# Local audio / music recommendations

Open **Audio_RAG_Music_Recommendations.ipynb** in local Jupyter, reload from disk, restart the kernel and run cells top to bottom. Assets are prepared on this laptop. CPU imports may take a minute.

## Completed pipeline
- WAV inspection and 32 kHz mono preprocessing.
- PANNs Cnn14: 2048-dimensional normalized embeddings, averaging up to three 10-second windows.
- Tiny AAM: 20 musical tracks, split into 16 references and 4 held-out queries.
- Local persistent ChromaDB, exact-file deduplication and self-match exclusion.
- Example-song queries, optional BPM/instrument filters, playback and grounded template explanations.
- Step 7B: user-triggered 20-second laptop microphone recording of phone-speaker audio, followed by five local recommendations. Never records automatically; microphone files stay under ignored data/.
- Held-out evaluation against a seeded random baseline using weak metadata proxies.

No Pinecone, Colab, generative LLM or free-form text-to-audio query is used. Uploads stay local. Similarity scores are not preference probabilities. Listen to results before drawing conclusions.

## Files
`audio_rag.py`: encoder/retriever. `catalogue.json`: portable metadata.
`evaluation.csv`, `verification.json`: actual run results. `lab_original/`: unchanged starter.
`setup_assets.py`: download and catalogue setup. Local weights, datasets and databases are excluded from Git.

## Fresh-machine reproduction
Use a standard Python environment compatible with requirements (this laptop runs Python 3.14).
1. Install PyTorch for your platform, then `python -m pip install -r requirements.txt`.
2. Run `python setup_assets.py` (about 500 MB download; allow over 1 GB free).
3. Start `python -m notebook Audio_RAG_Music_Recommendations.ipynb` and run the cells.

Normal installed packages are preferred. This laptop also uses project-local audio dependencies and shared Chroma dependencies from the image project. These fallbacks are unnecessary when requirements are installed normally. Model definitions are imported directly to avoid the legacy wrapper's home-folder writes and wget invocation.

## Attribution
Tiny AAM: Fabian Ostermann and Igor Vatolkin, 2022, https://doi.org/10.5281/zenodo.6771120, CC BY 4.0. Full-mix MP3s and beat annotations are used. Instrument metadata comes from stem names; tempo derives from beat timestamps. No titles, artists or genres are invented.
PANNs: Kong et al., https://github.com/qiuqiangkong/panns_inference (MIT), official checkpoint https://zenodo.org/records/3987831.
Original exercise: https://github.com/ironhack-labs/lab-music-recommendations.

## Before Git
Keep weights, audio, uploads, databases, packages, runtime/token files and .env out of Git; `.gitignore` is supplied. Human listening is still required to assess recommendation quality.

## Expanded recorder catalogue
`expand_catalogue.py` downloads selected 30-second MP3 excerpts from the official
FMA medium archive using validated HTTP byte ranges; the full archive is never
downloaded. It retains per-track licenses, artist/title and original genre tags
in `catalogue_fma.json`. Latin is a broad sampling group including Latin,
Latin America, Brazilian, Cumbia, Salsa and Tango tags. Run `--latin-only` to
supplement that group in a previously downloaded catalogue.
`build_expanded_index.py` creates a separate `fma_balanced_cnn14_v1` Chroma
collection, checks saved queries, activates it and serves the recorder.
`run_recorder.py` subsequently starts the saved active catalogue without
re-indexing. Step 7B also uses the active collection; the earlier Tiny AAM
teaching examples and their evaluation remain on their original collection.
FMA metadata: CC BY 4.0; audio licenses vary and are recorded per track.
Use this local exercise in accordance with those licenses; some are
noncommercial or prohibit redistribution of derivatives.
The same PANNs encoder is retained. Genre tags are reference metadata,
not a prediction or a guarantee of genre correctness.
