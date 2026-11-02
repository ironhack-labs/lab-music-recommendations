"""Local retrieval-augmented music recommendations, without API keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DEFAULT_SR = 22050
EMBEDDING_DIM = 53
FEATURE_VERSION = "mfcc20-chroma12-rms-v1-30sec"
DEFAULT_CATALOG = ROOT / "data/catalog.json"
DEFAULT_INDEX = ROOT / "data/music_index.npz"
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}


def extract_embedding(audio_path: str | Path, sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """MFCC mean/variance (40), chroma means (12), RMS (1), first 30 seconds.

    Uses SciPy, without neural checkpoints or first-run JIT compilation.
    Reject empty, silent, and non-finite signals instead of indexing them.
    """
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    with sf.SoundFile(audio_path) as audio:
        sr = audio.samplerate
        y = audio.read(frames=sr * 30, dtype="float32", always_2d=True).mean(axis=1)
    if not y.size or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError(f"Audio is empty, silent, or non-finite: {audio_path}")
    if sr != sample_rate:
        factor = gcd(sr, sample_rate)
        y = resample_poly(y, sample_rate // factor, sr // factor)
    y = np.pad(y, (0, max(0, 2048 - len(y))))
    frequencies, _, spectrum = stft(y, fs=sample_rate, nperseg=2048, noverlap=1536,
                                    boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    mel_max = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_max, 42) / 2595) - 1)
    bank = np.maximum(0, np.minimum(
        (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
        (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]))
    mfcc = dct(np.log(np.maximum(bank @ power, 1e-12)), axis=0, norm="ortho")[:20]
    chroma = np.zeros((12, power.shape[1]))
    valid = frequencies >= 27.5
    pitches = np.rint(69 + 12 * np.log2(frequencies[valid] / 440)).astype(int) % 12
    np.add.at(chroma, pitches, power[valid])
    chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), 1e-12)
    result = np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1),
                             chroma.mean(axis=1), [np.sqrt(np.mean(y ** 2))]])
    if result.shape != (EMBEDDING_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid audio features: {audio_path}")
    return result.astype(np.float32)


def cosine_similarity(vector_a, vector_b):
    a, b = np.asarray(vector_a), np.asarray(vector_b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Cosine similarity requires equal-sized 1D vectors")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator else 0.0


def _normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def _top_k(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k must be a positive integer")


def recommend(query_audio: str | Path, library_dir: str | Path, top_k: int = 5):
    """Compare real local files, excluding the query; never synthesize candidates."""
    _top_k(top_k)
    query, library = Path(query_audio).resolve(), Path(library_dir).resolve()
    if not query.is_file():
        raise FileNotFoundError(query)
    if not library.is_dir():
        raise NotADirectoryError(library)
    files = sorted(p for p in library.iterdir() if p.is_file()
                   and p.suffix.lower() in AUDIO_EXTENSIONS and p.resolve() != query)
    if not files:
        raise ValueError("No other audio tracks; add real music recordings")
    features = np.stack([extract_embedding(query)] + [extract_embedding(p) for p in files])
    scaled = _normalize((features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-6))
    return sorted(((float(v @ scaled[0]), p.name) for p, v in zip(files, scaled[1:])),
                  key=lambda x: (-x[0], x[1]))[:top_k]


def _document(track):
    return " ".join([track["title"], track["artist"], track["genre"],
                     track["description"], *track["tags"]])


def _catalog(path):
    tracks = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(tracks) < 2 or len({t["id"] for t in tracks}) != len(tracks):
        raise ValueError("Catalog needs at least two tracks with unique IDs")
    for track in tracks:
        for field in ("id", "title", "artist", "genre", "description", "tags", "path", "source"):
            if field not in track:
                raise ValueError(f"Catalog track missing {field}")
    return tracks


def _fingerprint(tracks, catalog_path):
    digest = hashlib.sha256((FEATURE_VERSION + json.dumps(tracks, sort_keys=True)).encode())
    for track in tracks:
        path = Path(catalog_path).parent / track["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run python scripts/download_music.py")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_index(catalog_path=DEFAULT_CATALOG, index_path=DEFAULT_INDEX):
    """Persist vectors and metadata; reuse only when catalog and audio are unchanged."""
    catalog_path, index_path = Path(catalog_path).resolve(), Path(index_path).resolve()
    tracks = _catalog(catalog_path)
    fingerprint = _fingerprint(tracks, catalog_path)
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                return LocalMusicRAG.load(index_path, catalog_path)
    features = np.stack([extract_embedding(catalog_path.parent / t["path"]) for t in tracks])
    mean, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-6)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_path, vectors=_normalize((features - mean) / scale),
                        mean=mean, scale=scale, tracks=json.dumps(tracks),
                        fingerprint=fingerprint, version=FEATURE_VERSION)
    return LocalMusicRAG.load(index_path, catalog_path)


class LocalMusicRAG:
    """TF-IDF/audio retrieval followed by grounded template generation (no LLM)."""
    @classmethod
    def load(cls, index_path=DEFAULT_INDEX, catalog_path=None):
        self = cls()
        with np.load(index_path, allow_pickle=False) as saved:
            if str(saved["version"]) != FEATURE_VERSION:
                raise ValueError("Index feature version changed; rebuild the index")
            self.tracks = json.loads(str(saved["tracks"]))
            self.vectors, self.mean, self.scale = saved["vectors"], saved["mean"], saved["scale"]
        # Catalog/audio paths stay relative, so the project can be moved to another machine.
        self.catalog_path = Path(catalog_path or Path(index_path).parent / "catalog.json").resolve()
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.text_vectors = self.vectorizer.fit_transform([_document(t) for t in self.tracks])
        return self

    def retrieve(self, preferences="", seed_ids=(), query_audio=None, top_k=5,
                 audio_weight=0.5, genre=None, exclude_ids=()):
        _top_k(top_k)
        if not 0 <= audio_weight <= 1:
            raise ValueError("audio_weight must be between 0 and 1")
        seed_ids = [seed_ids] if isinstance(seed_ids, str) else list(seed_ids)
        by_id = {t["id"]: i for i, t in enumerate(self.tracks)}
        unknown = set(seed_ids) - by_id.keys()
        if unknown:
            raise ValueError(f"Unknown seed IDs: {sorted(unknown)}")
        audio_queries = [self.vectors[by_id[s]] for s in seed_ids]
        excluded = set(exclude_ids) | set(seed_ids)
        if query_audio is not None:
            path = Path(query_audio).resolve()
            audio_queries.append(_normalize((extract_embedding(path) - self.mean) / self.scale))
            excluded.update(t["id"] for t in self.tracks
                            if (self.catalog_path.parent / t["path"]).resolve() == path)
        text_query = self.vectorizer.transform([preferences])
        has_text = bool(text_query.nnz)
        if not audio_queries and not has_text:
            raise ValueError("No recognized preferences; try piano, jazz, classical, vocals, or a seed ID")
        if preferences.strip() and not has_text:
            raise ValueError("Preference text has no catalog terms; remove it to use audio alone")
        text_scores = (self.text_vectors @ text_query.T).toarray().ravel()
        audio_scores = np.zeros(len(self.tracks))
        if audio_queries:
            audio_scores = self.vectors @ _normalize(np.mean(audio_queries, axis=0))
        weight = audio_weight if has_text and audio_queries else float(bool(audio_queries))
        scores = weight * (audio_scores + 1) / 2 + (1 - weight) * text_scores
        query_terms = set(self.vectorizer.build_analyzer()(preferences))
        results = []
        for i in np.argsort(-scores, kind="stable"):
            track = self.tracks[i]
            if track["id"] in excluded or (genre and track["genre"].casefold() != genre.casefold()):
                continue
            if not audio_queries and text_scores[i] <= 0:
                continue
            matched = sorted(query_terms & set(self.vectorizer.build_analyzer()(_document(track))))
            evidence = f"Catalog terms: {', '.join(matched)}." if matched else "Retrieved by seed/audio similarity."
            results.append({**track, "score": float(scores[i]), "text_score": float(text_scores[i]),
                            "audio_score": float(audio_scores[i]) if audio_queries else None,
                            "evidence": evidence})
            if len(results) == top_k:
                break
        return results

    @staticmethod
    def generate_answer(results):
        if not results:
            return "No tracks match these filters. Try a broader genre or preference."
        return "\n".join(
            f"{i}. {r['title']} — {r['artist']} [{r['id']}]\n"
            f"   {r['description']} {r['evidence']}\n"
            f"   Retrieval score: {r['score']:.3f}; source: {r['source']}"
            for i, r in enumerate(results, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build or reuse the local index")
    build.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query = sub.add_parser("query", help="Recommend using text, seed tracks, or audio")
    query.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query.add_argument("--preferences", default="")
    query.add_argument("--seed", action="append", default=[])
    query.add_argument("--audio", type=Path)
    query.add_argument("--genre")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--audio-weight", type=float, default=0.5)
    args = parser.parse_args()
    try:
        if args.command == "build":
            rag = build_index(args.catalog, args.index)
            print(f"Indexed {len(rag.tracks)} real recordings at {args.index}")
        else:
            rag = LocalMusicRAG.load(args.index)
            results = rag.retrieve(args.preferences, args.seed, args.audio,
                                   args.top_k, args.audio_weight, args.genre)
            print(rag.generate_answer(results))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
