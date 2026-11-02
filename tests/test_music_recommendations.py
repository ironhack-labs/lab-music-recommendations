import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from music_recommendations import (EMBEDDING_DIM, LocalMusicRAG, build_index,
                                   cosine_similarity, extract_embedding, recommend)


class MusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        tracks = []
        for i, (genre, tags) in enumerate([
            ('jazz', ['piano', 'relaxed']), ('electronic', ['drums', 'energetic']),
            ('classical', ['orchestra', 'strings']), ('folk', ['guitar', 'vocals'])]):
            t = np.arange(22050) / 22050
            y = 0.2 * np.sin(2 * np.pi * (220 + i * 170) * t)
            y += 0.08 * np.sin(2 * np.pi * (440 + i * 91) * t)
            path = cls.root / f'{i}.wav'
            sf.write(path, y, 22050)
            tracks.append(dict(id=str(i), title=f'Test {i}', artist='Test fixture',
                               genre=genre, tags=tags, description=' '.join(tags),
                               path=path.name, source='test fixture'))
        cls.catalog = cls.root / 'catalog.json'
        cls.catalog.write_text(json.dumps(tracks), encoding='utf-8')
        cls.index = cls.root / 'index.npz'
        cls.rag = build_index(cls.catalog, cls.index)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_finite_fixed_size_features_and_cosine(self):
        vector = extract_embedding(self.root / '0.wav')
        self.assertEqual(vector.shape, (EMBEDDING_DIM,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1, places=5)
        self.assertEqual(cosine_similarity(np.zeros(53), vector), 0)
        with self.assertRaises(ValueError):
            cosine_similarity(vector, np.zeros(40))

    def test_empty_silent_and_missing_audio(self):
        for name, y in [('empty.wav', np.array([])), ('silent.wav', np.zeros(4096))]:
            path = self.root / name
            sf.write(path, y, 22050)
            try:
                with self.assertRaisesRegex(ValueError, 'empty, silent'):
                    extract_embedding(path)
            finally:
                path.unlink()
        with self.assertRaises((OSError, sf.LibsndfileError)):
            extract_embedding(self.root / 'missing.wav')

    def test_preferences_filters_and_grounded_generation(self):
        results = self.rag.retrieve('relaxed piano')
        self.assertEqual(results[0]['id'], '0')
        self.assertEqual(len(results), 1)  # Do not pad with zero-match tracks.
        self.assertIn('piano', results[0]['evidence'])
        self.assertIn('[0]', self.rag.generate_answer(results))
        self.assertEqual(self.rag.retrieve('piano', genre='folk'), [])
        self.assertIn('No tracks', self.rag.generate_answer([]))

    def test_seed_and_absolute_query_excluded(self):
        results = self.rag.retrieve(seed_ids=['0'], top_k=20)
        self.assertEqual(len(results), 3)
        self.assertNotIn('0', [r['id'] for r in results])
        results = self.rag.retrieve(query_audio=(self.root / '0.wav').resolve())
        self.assertNotIn('0', [r['id'] for r in results])
        self.assertTrue(all(-1.00001 <= r['audio_score'] <= 1.00001 for r in results))
        results = recommend((self.root / '0.wav').resolve(), self.root, 3)
        self.assertNotIn('0.wav', [name for _, name in results])

    def test_invalid_requests(self):
        for top_k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.rag.retrieve('piano', top_k=top_k)
        for kwargs in ({}, {'preferences': 'zxqv'}, {'seed_ids': ['missing']},
                       {'preferences': 'piano', 'audio_weight': 2}):
            with self.assertRaises(ValueError):
                self.rag.retrieve(**kwargs)

    def test_hybrid_weight_endpoints(self):
        text = self.rag.retrieve('piano', seed_ids=['1'], audio_weight=0)[0]
        self.assertAlmostEqual(text['score'], text['text_score'])
        audio = self.rag.retrieve('piano', seed_ids=['1'], audio_weight=1)[0]
        self.assertAlmostEqual(audio['score'], (audio['audio_score'] + 1) / 2)

    def test_persistence_reuse_and_invalidation(self):
        original = self.index.stat().st_mtime_ns
        loaded = build_index(self.catalog, self.index)
        self.assertEqual(original, self.index.stat().st_mtime_ns)
        self.assertEqual(loaded.retrieve('piano'), self.rag.retrieve('piano'))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(self.catalog.read_text())
            for track in manifest:
                track['path'] = str(self.root / track['path'])
            catalog = Path(tmp) / 'catalog.json'
            index = Path(tmp) / 'index.npz'
            catalog.write_text(json.dumps(manifest))
            build_index(catalog, index)
            with np.load(index, allow_pickle=False) as saved:
                before = str(saved['fingerprint'])
            manifest[0]['tags'].append('newtag')
            catalog.write_text(json.dumps(manifest))
            updated = build_index(catalog, index)
            self.assertEqual(updated.retrieve('newtag')[0]['id'], '0')
            with np.load(index, allow_pickle=False) as saved:
                self.assertNotEqual(before, str(saved['fingerprint']))

    def test_single_file_library_is_not_synthesized(self):
        with tempfile.TemporaryDirectory() as tmp:
            query = Path(tmp) / 'only.wav'
            sf.write(query, np.ones(4096) * 0.1, 22050)
            with self.assertRaisesRegex(ValueError, 'No other audio'):
                recommend(query, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [query])


if __name__ == '__main__':
    unittest.main()
