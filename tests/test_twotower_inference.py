import contextlib
import unittest
from unittest.mock import patch

import torch

import improvnet.model.twotower_denoiser as denoiser_module
from improvnet.inference.twotower_inference import (
    FULL_POOL,
    TwoTowerInference,
    VocabularyConstraints,
    build_arg_parser,
    build_trajectory,
    constrained_top_predictions,
    select_reveal_positions,
    truncate_to_complete_events,
)
from improvnet.model.twotower_config import BLOCK_SIZE, MASK_ID
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.utils.ar_utils import ProcessData


class TwoTowerInferenceTests(unittest.TestCase):
    def _mock_inference(self):
        inference = object.__new__(TwoTowerInference)
        inference.device = torch.device("cpu")
        inference.processor = ProcessData()
        inference.tokenizer = inference.processor.tokenizer
        inference.constraints = VocabularyConstraints(inference.tokenizer)
        inference._tower_a_context = lambda committed, genre, multi_hot: (
            None,
            torch.ones(1, 2, dtype=torch.bool),
            2,
        )
        return inference

    def test_cli_defaults_to_jazz_full_mask_and_adas(self):
        args = build_arg_parser().parse_args(["--input", "example.mid"])
        self.assertEqual(args.genre, "jazz")
        self.assertEqual(args.strategy, "full-mask")
        self.assertEqual(args.sampler, "adas")
        self.assertEqual(args.revision_mode, "timing")

    def test_full_vocabulary_pool_excludes_internal_and_prefix_tokens(self):
        tokenizer = AbsTokenizer()
        constraints = VocabularyConstraints(tokenizer)
        allowed = set(constraints.pools[FULL_POOL])

        for token in ("<S>", "<P>", "<U>", "<MASK>", "<BLANK>", "<SEP>"):
            self.assertNotIn(tokenizer.tok_to_id[token], allowed)
        for token in ("<E>", "<D>", "<T>"):
            self.assertIn(tokenizer.tok_to_id[token], allowed)
        self.assertFalse(
            any(
                isinstance(tokenizer.id_to_tok[token_id], tuple)
                and tokenizer.id_to_tok[token_id][0] == "prefix"
                for token_id in allowed
            )
        )

    def test_constrained_prediction_uses_only_requested_token_type(self):
        tokenizer = AbsTokenizer()
        constraints = VocabularyConstraints(tokenizer)
        logits = torch.zeros(2, tokenizer.vocab_size)
        note_id = constraints.pools["note"][0]
        onset_id = constraints.pools["onset"][0]
        logits[0, note_id] = 100.0
        logits[0, onset_id] = 10.0
        logits[1, onset_id] = 100.0
        logits[1, note_id] = 10.0

        predictions, confidence, _ = constrained_top_predictions(
            logits, ["onset", "note"], constraints
        )
        self.assertNotEqual(predictions[0].item(), note_id)
        self.assertNotEqual(predictions[1].item(), onset_id)
        self.assertGreater(confidence.min().item(), 0.0)

    def test_adas_discounts_a_candidate_attending_to_first_selection(self):
        confidence = torch.tensor([0.90, 0.85, 0.80])
        candidates = torch.ones(3, dtype=torch.bool)
        attention = torch.zeros(3, 3)
        attention[1, 0] = 1.0

        confidence_order = select_reveal_positions(
            confidence, candidates, 2, sampler="confidence"
        )
        adas_order = select_reveal_positions(
            confidence,
            candidates,
            2,
            sampler="adas",
            attention=attention,
            alpha=1.0,
        )

        self.assertEqual(confidence_order, [0, 1])
        self.assertEqual(adas_order, [0, 2])

    def test_trajectory_matches_training_staircase_lengths_and_padding_masks(self):
        valid_mask = torch.ones(BLOCK_SIZE, dtype=torch.bool)
        valid_mask[-3:] = False
        snapshots = [torch.full((BLOCK_SIZE,), MASK_ID) for _ in range(4)]

        for count, expected_length in ((1, 256), (2, 513), (3, 770), (4, 1027)):
            trajectory, timesteps, attention_mask = build_trajectory(
                snapshots[:count], [1.0] * count, valid_mask
            )
            self.assertEqual(trajectory.numel(), expected_length)
            self.assertEqual(timesteps.numel(), expected_length)
            self.assertEqual(attention_mask.numel(), expected_length)
            self.assertEqual(attention_mask[-3:].tolist(), [False, False, False])

    def test_complete_event_truncation_drops_only_partial_tail(self):
        tokens = [
            "<S>",
            ("instrument", "Acoustic Piano"),
            ("note", 60, 80),
            ("onset", 0),
            ("dur", 100),
            "<T>",
            ("instrument", "Acoustic Piano"),
            ("note", 64, 80),
        ]
        self.assertEqual(truncate_to_complete_events(tokens), tokens[:6])

    def test_full_mask_generation_ignores_source_after_prompt(self):
        inference = self._mock_inference()
        source = [
            "<S>",
            ("instrument", "Acoustic Piano"),
            ("note", 60, 80),
            ("onset", 0),
            ("dur", 100),
            "<T>",
            ("instrument", "Acoustic Piano"),
            ("note", 64, 80),
            ("onset", 0),
            ("dur", 100),
            "<E>",
        ]
        dim_id = inference.tokenizer.tok_to_id["<D>"]
        eos_id = inference.tokenizer.tok_to_id["<E>"]

        def decode(**kwargs):
            self.assertEqual(kwargs["strategy"], "full-mask")
            self.assertTrue(torch.all(kwargs["initial_state"][:2] == MASK_ID))
            result = kwargs["initial_state"].clone()
            result[0] = dim_id
            result[1] = eos_id
            return result, 0

        inference._decode_block = decode
        generated = inference.generate(source, prompt_len=5, max_tokens=7)
        self.assertEqual(generated, source[:5] + ["<D>", "<E>"])

    def test_onset_duration_generation_masks_only_timing_slots(self):
        inference = self._mock_inference()
        source = [
            "<S>",
            ("instrument", "Acoustic Piano"),
            ("note", 60, 80),
            ("onset", 0),
            ("dur", 100),
            "<E>",
        ]
        source_ids = inference.processor.tokens_to_tensor(source)

        def decode(**kwargs):
            self.assertEqual(kwargs["strategy"], "onset-duration")
            initial = kwargs["initial_state"]
            self.assertEqual(initial[0].item(), source_ids[1].item())
            self.assertEqual(initial[1].item(), source_ids[2].item())
            self.assertEqual(initial[2].item(), MASK_ID)
            self.assertEqual(initial[3].item(), MASK_ID)
            self.assertEqual(initial[4].item(), source_ids[5].item())
            result = initial.clone()
            result[:5] = source_ids[1:]
            return result, 0

        inference._decode_block = decode
        generated = inference.generate(
            source,
            strategy="onset-duration",
            prompt_len=1,
            max_tokens=6,
        )
        self.assertEqual(generated, source)

    def test_attention_capture_preserves_logits_and_masks_padding(self):
        replacements = {
            "VOCAB_SIZE": 32,
            "EMBED_DIM": 32,
            "N_HEADS": 4,
            "N_KV_HEADS": 2,
            "N_LAYERS": 2,
            "NUM_GENRES": 4,
            "NUM_INSTRUMENTS": 3,
        }
        with contextlib.ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch.object(denoiser_module, name, value))

            torch.manual_seed(7)
            model = denoiser_module.TwoTowerDenoiser().eval()
            tokens = torch.randint(0, 32, (1, 9))
            timesteps = torch.full((1, 9), 0.5)
            attention_mask = torch.ones(1, 9, dtype=torch.bool)
            attention_mask[0, -1] = False
            context_mask = torch.ones(1, 2, dtype=torch.bool)
            context = [
                (
                    torch.randn(1, 2, 2, 8),
                    torch.randn(1, 2, 2, 8),
                )
                for _ in range(2)
            ]
            controls = {
                "elasticity": torch.zeros(1),
                "genre": torch.zeros(1, dtype=torch.long),
                "multi_hot": torch.zeros(1, 3),
            }

            with torch.no_grad():
                normal_logits = model(
                    tokens,
                    timesteps,
                    context_kv_cache=context,
                    attention_mask=attention_mask,
                    context_attention_mask=context_mask,
                    draft_size=5,
                    **controls,
                )
                captured_logits, attention = model(
                    tokens,
                    timesteps,
                    context_kv_cache=context,
                    attention_mask=attention_mask,
                    context_attention_mask=context_mask,
                    draft_size=5,
                    return_last_attention=True,
                    **controls,
                )

            torch.testing.assert_close(
                captured_logits, normal_logits, atol=2e-5, rtol=2e-5
            )
            self.assertEqual(attention.shape, (1, 4, 11))
            self.assertEqual(torch.count_nonzero(attention[..., -1]).item(), 0)


if __name__ == "__main__":
    unittest.main()
