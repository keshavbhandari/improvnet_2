"""Analytical, exhaustive and adversarial pitch-retention tests; no GPU/model."""

from dataclasses import replace
from itertools import product
import math
import random
import unittest

import numpy as np

from improvnet.metrics.pitch_similarity import PitchSimilarityConfig, PitchSimilarityMetric
from improvnet.metrics.pitch_similarity.metric import align_cross_similarity


class PitchSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.metric = PitchSimilarityMetric()
        self.reference = [60, 62, 64, 67, 65, 64, 62, 60]

    def test_exact_copy_all_components(self):
        result = self.metric.evaluate_pitches(self.reference, self.reference)
        for value in (result.score, result.pitch_identity, result.pattern_preservation, result.structural_preservation,
                      result.reference_coverage, result.generated_precision, result.ssm_cosine):
            self.assertAlmostEqual(value, 1, places=12)
        self.assertEqual(result.alignment, tuple((i, i) for i in range(len(self.reference))))

    def test_feature_kernel_matches_explicit_cosine_features(self):
        c = self.metric.config
        d = np.abs(np.arange(12)[:, None] - np.arange(12)[None])
        d = np.minimum(d, 12-d)
        smooth = np.exp(-.5*(d/c.smooth_chroma_sigma)**2)
        smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
        pitches = [0, 60, 61, 72, 127]
        features = np.asarray([np.concatenate((math.sqrt(c.exact_pitch_weight)*np.eye(128)[p],
                                             math.sqrt(c.pitch_class_weight)*np.eye(12)[p % 12],
                                             math.sqrt(c.smooth_chroma_weight)*smooth[p % 12])) for p in pitches])
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        np.testing.assert_allclose(self.metric.cross_similarity(pitches, pitches), features@features.T, atol=1e-12)

    def test_kernel_is_symmetric_psd_and_octave_tolerant(self):
        kernel = self.metric.cross_similarity(range(128), range(128))
        np.testing.assert_allclose(kernel, kernel.T)
        np.testing.assert_allclose(np.diag(kernel), 1)
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-12)
        self.assertAlmostEqual(kernel[60, 72], .85, places=10)
        self.assertTrue(0 < kernel[60, 61] < kernel[60, 72] < kernel[60, 60])

    def test_raw_scalar_cosine_is_not_used(self):
        self.assertLess(self.metric.evaluate_pitches([60], [66]).pitch_identity, .01)

    def test_ornaments_preserve_gapped_patterns_and_structure(self):
        result = self.metric.evaluate_pitches([60, 62, 64, 67], [60, 61, 62, 63, 64, 66, 67])
        self.assertEqual(result.alignment, ((0, 0), (1, 2), (2, 4), (3, 6)))
        self.assertAlmostEqual(result.pitch_identity, 20/23)
        self.assertAlmostEqual(result.pattern_preservation, 1)
        self.assertAlmostEqual(result.structural_preservation, 1)
        self.assertAlmostEqual(result.reference_coverage, 1)
        self.assertLess(result.generated_precision, 1)

    def test_pattern_insertions_are_bounded(self):
        metric = PitchSimilarityMetric(replace(self.metric.config, max_pattern_insertions=0))
        result = metric.evaluate_pitches([60, 62, 64, 67], [60, 61, 62, 63, 64, 66, 67])
        self.assertEqual(result.pattern_preservation, 0)

    def test_deletions_cannot_receive_full_reference_coverage(self):
        result = self.metric.evaluate_pitches(self.reference, self.reference[::2])
        self.assertLess(result.reference_coverage, .8)
        self.assertLess(result.pitch_identity, .8)
        self.assertLess(result.pattern_preservation, 1)

    def test_substitution_and_randomization_ordering(self):
        substituted = list(self.reference)
        substituted[3] += 1
        copy = self.metric.evaluate_pitches(self.reference, self.reference)
        changed = self.metric.evaluate_pitches(self.reference, substituted)
        unrelated = self.metric.evaluate_pitches(self.reference, [66, 70, 73, 75, 78, 82, 85, 87])
        self.assertGreater(copy.score, changed.score)
        self.assertGreater(changed.score, unrelated.score)

    def test_shuffle_with_identical_pitch_histogram_is_not_perfect(self):
        shuffled = list(reversed(self.reference))
        # This particular reference is palindromic in part; use a non-palindrome.
        original = [60, 62, 64, 65, 67, 69, 71, 72]
        shuffled = original[::-1]
        result = self.metric.evaluate_pitches(original, shuffled)
        self.assertLess(result.score, .7)
        self.assertLess(result.pattern_preservation, .7)

    def test_ssm_identity_matrix_failure_does_not_reward_unrelated_pitches(self):
        metric = PitchSimilarityMetric(replace(self.metric.config, exact_pitch_weight=1, pitch_class_weight=0, smooth_chroma_weight=0))
        a, b = [60, 62, 64, 67], [61, 63, 66, 70]
        np.testing.assert_allclose(metric.cross_similarity(a, a), metric.cross_similarity(b, b))
        self.assertEqual(metric.evaluate_pitches(a, b).score, 0)
        same = metric.evaluate_pitches(a, a)
        self.assertIsNone(same.ssm_cosine)
        self.assertIsNone(same.structural_preservation)
        self.assertAlmostEqual(same.score, 1)

    def test_ssm_diagonal_exclusion_coverage_and_hand_cosine(self):
        original = [60, 64, 60, 65, 60, 67]
        result = self.metric.evaluate_pitches(original, original[:3])
        self.assertAlmostEqual(result.ssm_cosine, 1)
        self.assertAlmostEqual(result.structural_pair_coverage, 3/15)
        self.assertAlmostEqual(result.structural_preservation, .2)
        a, b = [60, 64, 67], [60, 65, 67]
        cross = self.metric.cross_similarity(a, b)
        pairs = ((0, 0), (1, 1), (2, 2))
        value, cosine, coverage = self.metric._structure(np.array(a), np.array(b), pairs, cross)
        upper = np.triu_indices(3, 1)
        weights = np.outer(np.diag(cross), np.diag(cross))[upper]
        x = self.metric.cross_similarity(a, a)[upper] * np.sqrt(weights)
        y = self.metric.cross_similarity(b, b)[upper] * np.sqrt(weights)
        self.assertAlmostEqual(cosine, float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))))
        self.assertAlmostEqual(value, cosine*coverage)

    def test_tiny_fragment_and_excessive_junk_are_not_high_scores(self):
        self.assertLess(self.metric.evaluate_pitches(self.reference, [60]).score, .2)
        result = self.metric.evaluate_pitches(self.reference, self.reference + [66]*200)
        self.assertLess(result.score, .5)

    def test_global_transposition_and_virtual_range_edges(self):
        original = [0, 2, 4, 7, 5, 2]
        generated = [p+7 for p in original]
        metric = PitchSimilarityMetric(replace(self.metric.config, allow_global_transposition=True))
        result = metric.evaluate_pitches(original, generated)
        self.assertAlmostEqual(result.score, 1)
        self.assertEqual(result.generated_to_reference_transpose, -7)
        self.assertLess(self.metric.evaluate_pitches(original, generated).score, 1)
        self.assertLess(metric.cross_similarity([0], [0], transpose=-12)[0,0], 1)

    def test_singleton_short_and_empty_inputs(self):
        result = self.metric.evaluate_pitches([60], [60])
        self.assertAlmostEqual(result.score, 1)
        self.assertIsNone(result.pattern_preservation)
        self.assertIsNone(result.structural_preservation)
        for reference in ([60], self.reference):
            result = self.metric.evaluate_pitches(reference, [])
            self.assertEqual((result.score, result.pitch_identity, result.generated_notes), (0,0,0))
        with self.assertRaises(ValueError):
            self.metric.evaluate_pitches([], self.reference)

    def test_repeated_pitch_reference_can_be_preserved_without_note_reuse(self):
        result = self.metric.evaluate_pitches([60]*8, [60]*8)
        self.assertAlmostEqual(result.score, 1)
        short = self.metric.evaluate_pitches([60]*8, [60]*2)
        self.assertEqual(short.matched_notes, 2)
        self.assertLess(short.score, .4)

    def test_randomized_bounds_and_monotone_deterministic_alignment(self):
        rng = random.Random(7)
        for _ in range(60):
            a, b = [[rng.randrange(48,84) for _ in range(rng.randrange(1,14))] for _ in range(2)]
            result = self.metric.evaluate_pitches(a, b)
            self.assertEqual(result, self.metric.evaluate_pitches(a, b))
            for value in (result.score, result.pitch_identity, result.pattern_preservation, result.structural_preservation,
                          result.reference_coverage, result.generated_precision, result.ssm_cosine, result.structural_pair_coverage):
                if value is not None:
                    self.assertTrue(math.isfinite(value) and -1e-12 <= value <= 1+1e-12)
            self.assertLessEqual(result.matched_notes, min(len(a),len(b)))
            self.assertTrue(all(i<ii and j<jj for (i,j),(ii,jj) in zip(result.alignment,result.alignment[1:])))

    def test_affine_alignment_matches_exhaustive_tiny_path_search(self):
        c = self.metric.config
        def exhaustive(cross, i=0, j=0, previous=0):
            n,m = cross.shape
            if i==n and j==m:
                return 0.
            options = []
            if i<n and j<m:
                options.append(cross[i,j]-c.match_baseline+exhaustive(cross,i+1,j+1,0))
            if i<n:
                options.append(-(c.delete_extend if previous==1 else c.delete_open)+exhaustive(cross,i+1,j,1))
            if j<m:
                options.append(-(c.insert_extend if previous==2 else c.insert_open)+exhaustive(cross,i,j+1,2))
            return max(options)
        rng = np.random.default_rng(8)
        for n,m in product(range(4),repeat=2):
            for _ in range(4):
                cross = rng.random((n,m))
                result = align_cross_similarity(cross,c)
                self.assertAlmostEqual(result.objective,exhaustive(cross),places=12)

    def test_invalid_pitches_config_matrices_and_note_limits(self):
        for values in ([True],[60.],[128],[-1],['60'],[[60]],np.zeros((2,2)),[float('nan')]):
            with self.subTest(values=str(values)), self.assertRaises(ValueError):
                self.metric.evaluate_pitches(self.reference,values)
        for kwargs in ({"exact_pitch_weight":-1},{"smooth_chroma_sigma":0},{"retention_beta":float('inf')},
                       {"min_match_similarity":True},{"identity_weight":0},{"pattern_lengths":(1,)},
                       {"pattern_lengths":(3,3)},{"max_notes":False},{"allow_global_transposition":1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PitchSimilarityConfig(**kwargs)
        for matrix in ([1,2],[[float('nan')]],[[1.1]],[[-.1]]):
            with self.assertRaises(ValueError):
                align_cross_similarity(matrix,self.metric.config)
        limited = PitchSimilarityMetric(replace(self.metric.config,max_notes=2))
        with self.assertRaises(ValueError):
            limited.evaluate_pitches(self.reference,self.reference)


class TokenPitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from improvnet.tokenizer.absolute import AbsTokenizer
        cls.tokenizer = AbsTokenizer()

    def setUp(self):
        self.metric = PitchSimilarityMetric(tokenizer=self.tokenizer)

    def event(self,pitch,instrument="Acoustic Piano",onset=0,duration=100,velocity=80):
        return [("instrument",instrument),("note",pitch,velocity),("onset",onset),("dur",duration)]

    def test_timing_velocity_metadata_and_drums_do_not_enter_score(self):
        a = ['<S>']+sum((self.event(p) for p in (60,62,64,67)),[])+['<E>']
        b = ['<S>']+sum((self.event(p,onset=200,duration=200,velocity=60)+self.event(36,'drum')+['<T>'] for p in (60,62,64,67)),[])+['<D>','<E>']
        self.assertAlmostEqual(self.metric.evaluate_tokens(a,b).score,1)
        prefix = self.tokenizer.prefix_tokens[0]
        self.assertAlmostEqual(self.metric.evaluate_tokens([prefix]+a,b).score,1)

    def test_context_resolves_mid_event_drums_without_scoring_history(self):
        canvas = [('note',36,80),('onset',0),('dur',100)]+self.event(60)
        history = self.event(67)+[('instrument','drum')]
        self.assertEqual(self.metric.pitches_from_tokens(canvas,context=history),[60])

    def test_ids_padding_masks_and_unresolved_tokens(self):
        tokens = self.event(60)+self.event(62)+self.event(64)
        ids = [self.tokenizer.tok_to_id[t] for t in tokens]
        padded = ids+[self.tokenizer.pad_id]*4
        self.assertAlmostEqual(self.metric.evaluate_tokens(ids,padded,generated_mask=[True]*len(ids)+[False]*4).score,1)
        for invalid in (padded,['<MASK>'],[('note',60.,80)],[True],[["bad"]],[999999]):
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                self.metric.evaluate_tokens(tokens,invalid)
        with self.assertRaises(ValueError):
            self.metric.evaluate_tokens(ids,padded,generated_mask=[True])

    def test_empty_terminal_only_and_drum_only_outputs(self):
        original = self.event(60)
        for generated in ([],['<T>','<E>'],self.event(36,'drum')):
            self.assertEqual(self.metric.evaluate_tokens(original,generated).score,0)
        with self.assertRaises(ValueError):
            self.metric.evaluate_tokens(self.event(36,'drum'),original)


if __name__ == '__main__':
    unittest.main()
