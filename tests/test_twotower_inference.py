import contextlib
import unittest
from unittest.mock import patch

import torch

import improvnet.model.twotower_denoiser as denoiser_module
from improvnet.inference.twotower_inference import (
    ELASTIC_POOL,
    ELASTIC_SLOT_PREFIX,
    FULL_POOL,
    TwoTowerInference,
    VocabularyConstraints,
    build_arg_parser,
    build_trajectory,
    constrained_top_predictions,
    constrained_predictions,
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
        self.assertEqual(args.refinement_passes, 1)
        self.assertEqual(args.elasticity, 0.0)
        self.assertEqual(args.temperature, 1.0)
        self.assertIsNone(args.seed)

    def test_sampling_temperature_seed_and_sampled_token_confidence(self):
        constraints = VocabularyConstraints(AbsTokenizer())
        # Two outcomes with a known categorical distribution; repeated rows
        # check actual sampling rather than merely checking a code path.
        constraints.pools["test"] = [0, 1]
        logits = torch.tensor([1.0, 0.0]).repeat(4000, 1)
        def draw(temperature):
            return constrained_predictions(
                logits, ["test"] * len(logits), constraints,
                sampling_temperature=temperature,
                generator=torch.Generator().manual_seed(71),
            )
        greedy, confidence, _, _ = draw(0.0)
        sampled, sampled_confidence, _, proposal_probability = draw(1.0)
        hot, _, _, _ = draw(2.0)
        self.assertTrue(torch.all(greedy == 0))
        self.assertTrue(torch.equal(sampled, draw(1.0)[0]))
        self.assertAlmostEqual((sampled == 0).float().mean().item(), 0.731, delta=0.03)
        self.assertAlmostEqual((hot == 0).float().mean().item(), 0.622, delta=0.03)
        torch.testing.assert_close(proposal_probability, sampled_confidence)
        self.assertAlmostEqual(sampled_confidence[sampled == 1][0].item(), 0.269, delta=0.001)
        self.assertTrue(torch.all(proposal_probability[sampled == 1] < confidence[sampled == 1]))

    def test_timing_revision_and_remasking_exclude_structure_and_blanks(self):
        inference = self._mock_inference()
        tokens = ["<T>", "<E>", "<BLANK>", ("instrument", "Acoustic Piano"),
                  ("onset", 0), ("dur", 100)]
        state = torch.full((BLOCK_SIZE,), 2)
        state[:len(tokens)] = inference.processor.tokens_to_tensor(tokens)
        valid = torch.arange(BLOCK_SIZE) < len(tokens)
        roles = [FULL_POOL] * BLOCK_SIZE
        categories = inference._prediction_categories(state, valid, roles, "full-mask", "timing", 0.15)
        self.assertEqual(categories[:6], [None, None, None, None, "onset", "dur"])
        candidates = inference._refinement_candidates(state, valid, roles, "full-mask", "timing")
        self.assertEqual(candidates.nonzero().flatten().tolist(), [4, 5])
        all_candidates = inference._refinement_candidates(state, valid, roles, "full-mask", "all")
        self.assertEqual(all_candidates.nonzero().flatten().tolist(), list(range(6)))

    def test_sampled_adas_decode_resolves_masks_and_keeps_padding(self):
        inference = self._mock_inference()
        inference._autocast = contextlib.nullcontext
        token_ids = [inference.tokenizer.tok_to_id[token] for token in ("<T>", "<D>")]
        inference.constraints.pools[FULL_POOL] = token_ids
        lengths = []
        def model(**kwargs):
            length = kwargs["noisy_target"].shape[1]
            lengths.append(length)
            logits = torch.full((1, length, inference.tokenizer.vocab_size), -100.0)
            logits[:, :, token_ids] = 0.0
            attention = torch.full((1, BLOCK_SIZE, 2 + length), 1.0 / (2 + length))
            return (logits, attention) if kwargs["return_last_attention"] else logits
        inference.model_denoiser = model
        state = torch.full((BLOCK_SIZE,), 2)
        state[:16] = MASK_ID
        result, revisions, _ = inference._decode_block(
            initial_state=state, valid_mask=torch.arange(BLOCK_SIZE) < 16,
            source_roles=[FULL_POOL] * BLOCK_SIZE, strategy="full-mask",
            sampler="adas", adas_alpha=40.0, revision_mode="freeze",
            revision_margin=0.1, confidence_temperature=1.0,
            context_kv=None, context_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            seq_offset=2, genre=torch.zeros(1, dtype=torch.long), multi_hot=torch.zeros(1, 41),
            temperature=1.0, generator=torch.Generator().manual_seed(19),
        )
        self.assertEqual(lengths, [256, 513, 770, 1027])
        self.assertEqual(set(result[:16].tolist()), set(token_ids))
        self.assertTrue(torch.all(result[16:] == 2))
        self.assertEqual(revisions, 0)

    def test_adas_accumulates_directional_penalties_and_excludes_padding(self):
        confidence = torch.tensor([0.9, 0.85, 0.8, 0.75, 1.0])
        attention = torch.zeros(5, 5)
        attention[1, 0] = 1.0  # after selecting 0: scores 1=.75, 2=.8, 3=.75
        attention[1, 2] = 0.5  # after selecting 2: score 1=.65, 3=.75
        attention[0, 3] = 0.9  # reverse direction must not discount 3
        candidates = torch.tensor([True, True, True, True, False])
        self.assertEqual(select_reveal_positions(confidence, candidates, 3, "adas", attention, 1.0), [0, 2, 3])
        self.assertEqual(select_reveal_positions(confidence, candidates, 3, "adas", attention, 0.0), [0, 1, 2])

    def test_full_vocabulary_pool_excludes_internal_and_prefix_tokens(self):
        tokenizer = AbsTokenizer()
        constraints = VocabularyConstraints(tokenizer)
        allowed = set(constraints.pools[FULL_POOL])

        for token in ("<S>", "<P>", "<U>", "<MASK>", "<BLANK>", "<SEP>"):
            self.assertNotIn(tokenizer.tok_to_id[token], allowed)
        self.assertIn(tokenizer.tok_to_id["<BLANK>"], constraints.pools[ELASTIC_POOL])
        self.assertIn(
            tokenizer.tok_to_id["<BLANK>"],
            constraints.pools[f"{ELASTIC_SLOT_PREFIX}instrument"],
        )
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
            return result, 0, torch.ones(BLOCK_SIZE)

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
            return result, 0, torch.ones(BLOCK_SIZE)

        inference._decode_block = decode
        generated = inference.generate(
            source,
            strategy="onset-duration",
            prompt_len=1,
            max_tokens=6,
        )
        self.assertEqual(generated, source)

    def test_full_mask_elasticity_passes_condition_and_discards_blanks(self):
        inference = self._mock_inference()
        source = ["<S>", ("instrument", "Acoustic Piano"), ("note", 60, 80),
                  ("onset", 0), ("dur", 100)]
        blank_id = inference.tokenizer.tok_to_id["<BLANK>"]
        eos_id = inference.tokenizer.tok_to_id["<E>"]
        dim_id = inference.tokenizer.tok_to_id["<D>"]

        def decode(**kwargs):
            self.assertEqual(kwargs["elasticity"], 0.15)
            categories = inference._prediction_categories(
                kwargs["initial_state"], kwargs["valid_mask"], kwargs["source_roles"],
                kwargs["strategy"], kwargs["revision_mode"], kwargs["elasticity"],
            )
            self.assertEqual(categories[:6], [ELASTIC_POOL] * 6)
            result = kwargs["initial_state"].clone()
            result[:6] = torch.tensor([blank_id] * 4 + [dim_id, eos_id])
            return result, 0, torch.ones(BLOCK_SIZE)

        inference._decode_block = decode
        self.assertEqual(
            inference.generate(source, prompt_len=5, max_tokens=11, elasticity=0.15),
            source + ["<D>", "<E>"],
        )

    def test_onset_duration_elastic_slots_preserve_source_and_group_blanks(self):
        inference = self._mock_inference()
        event = [("instrument", "Acoustic Piano"), ("note", 60, 80),
                 ("onset", 0), ("dur", 100)]
        source = ["<S>"] + event * 30 + ["<E>"]
        source_ids = inference.processor.tokens_to_tensor(source[1:]).tolist()
        blank_id = inference.tokenizer.tok_to_id["<BLANK>"]
        calls = []

        def decode(**kwargs):
            calls.append(kwargs)
            roles = kwargs["source_roles"]
            slot_positions = [i for i, role in enumerate(roles)
                              if role.startswith(ELASTIC_SLOT_PREFIX)]
            self.assertTrue(slot_positions)
            self.assertEqual(len(slot_positions) % 4, 0)
            self.assertAlmostEqual(
                kwargs["elasticity"], len(slot_positions) / kwargs["valid_mask"].sum().item()
            )
            categories = inference._prediction_categories(
                kwargs["initial_state"], kwargs["valid_mask"], roles,
                kwargs["strategy"], kwargs["revision_mode"], kwargs["elasticity"],
            )
            self.assertEqual(
                [categories[position] for position in slot_positions[:4]],
                [f"{ELASTIC_SLOT_PREFIX}{field}" for field in
                 ("instrument", "note", "onset", "dur")],
            )
            result = kwargs["initial_state"].clone()
            source_index = 0
            for position in range(kwargs["valid_mask"].sum().item()):
                if position in slot_positions:
                    result[position] = blank_id
                else:
                    result[position] = source_ids[source_index]
                    source_index += 1
            # A partly filled insertion group is discarded as a unit, so it
            # cannot leave malformed note fields in the MIDI continuation.
            result[slot_positions[0]] = source_ids[0]
            return result, 0, torch.ones(BLOCK_SIZE)

        inference._decode_block = decode
        generated = inference.generate(
            source, strategy="onset-duration", prompt_len=1,
            max_tokens=len(source) + 20, elasticity=0.15,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(generated, source)

    def test_elasticity_rejects_untrained_fraction(self):
        inference = self._mock_inference()
        with self.assertRaisesRegex(ValueError, "elasticity must be"):
            inference.generate(["<S>", "<T>"], max_tokens=2, elasticity=0.05)

    def test_iterative_refinement_restarts_from_previous_final_draft(self):
        inference = self._mock_inference()
        source = [
            "<S>",
            ("instrument", "Acoustic Piano"),
            ("note", 60, 80),
            ("onset", 0),
            ("dur", 100),
            "<E>",
        ]
        onset_ids = inference.constraints.pools["onset"][:4]
        calls = []

        def decode(**kwargs):
            initial = kwargs["initial_state"].clone()
            calls.append(initial)
            result = initial.clone()
            for position in range(4):
                if result[position].item() == MASK_ID:
                    result[position] = onset_ids[position]
            confidence = torch.ones(BLOCK_SIZE)
            confidence[:4] = torch.tensor([0.1, 0.2, 0.3, 0.4])
            return result, 0, confidence

        inference._decode_block = decode
        generated = inference.generate(
            source,
            prompt_len=5,
            max_tokens=9,
            refinement_passes=3,
            revision_mode="timing",
        )

        self.assertEqual(len(calls), 3)
        self.assertTrue(torch.all(calls[0][:4] == MASK_ID))
        # Passes two and three use noise levels 2/16 and 1/16. For a
        # four-token partial block both schedules re-mask one least-confident
        # editable position, while retaining the other final-draft outputs.
        self.assertEqual((calls[1][:4] == MASK_ID).sum().item(), 1)
        self.assertEqual((calls[2][:4] == MASK_ID).sum().item(), 1)
        self.assertEqual(generated[:5], source[:5])

    def test_iterative_refinement_rejects_frozen_outputs(self):
        inference = self._mock_inference()
        with self.assertRaisesRegex(ValueError, "requires revision_mode"):
            inference.generate(
                ["<S>", "<T>"],
                prompt_len=1,
                max_tokens=2,
                refinement_passes=2,
                revision_mode="freeze",
            )

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
