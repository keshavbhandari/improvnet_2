import contextlib
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import torch

import improvnet.model.ar_model as ar_module
from improvnet.model.uniform_diffusion_config import UniformDiffusionConfig
from improvnet.model.uniform_diffusion_model import UniformDiffusionModel


class UniformDiffusionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = UniformDiffusionConfig(
            vocab_size=32, embed_dim=32, n_heads=4, n_kv_heads=2,
            n_layers=2, num_genres=4, num_instruments=3, canvas_size=8,
            dropout=0.0, gradient_checkpointing=False,
        )
        self.model = UniformDiffusionModel(self.config).eval()
        self.genre = torch.tensor([1, 2])
        self.multi_hot = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        self.prompt = torch.randint(0, 32, (2, 5))
        self.canvas = torch.randint(0, 32, (2, 8))

    def _ar_model(self):
        c = self.config
        overrides = dict(VOCAB_SIZE=c.vocab_size, EMBED_DIM=c.embed_dim,
                         N_HEADS=c.n_heads, N_KV_HEADS=c.n_kv_heads, N_LAYERS=c.n_layers,
                         NUM_GENRES=c.num_genres, NUM_INSTRUMENTS=c.num_instruments,
                         HAS_FLASH=False)
        with contextlib.ExitStack() as stack:
            for name, value in overrides.items():
                stack.enter_context(patch.object(ar_module, name, value))
            return ar_module.ARContextModel().eval()

    def _context(self, **kwargs):
        return self.model.encode_context(self.prompt, genre=self.genre, multi_hot=self.multi_hot, **kwargs)

    def test_ar_checkpoint_causal_logits_and_every_cache_layer_match(self):
        ar = self._ar_model()
        source_weight = ar.token_emb.weight.detach().clone()
        report = self.model.initialize_from_ar({
            "model_state_dict": {"module." + key: value for key, value in ar.state_dict().items()}
        })
        self.assertTrue(all(key.startswith(("self_conditioning.", "elasticity_emb.")) for key in report.missing_keys))
        self.assertFalse(report.unexpected_keys)
        expected, expected_kv = ar(self.prompt, self.genre, self.multi_hot, use_cache=True)
        actual = self._context()
        torch.testing.assert_close(actual.logits, expected, atol=2e-6, rtol=2e-5)
        for got, wanted in zip(actual.cache.key_values, expected_kv):
            torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-5)
        self.assertNotEqual(ar.token_emb.weight.data_ptr(), self.model.token_emb.weight.data_ptr())
        with torch.no_grad():
            self.model.token_emb.weight.add_(1.0)
        torch.testing.assert_close(ar.token_emb.weight, source_weight)

    def test_checkpoint_loader_roundtrip_and_incompatible_weights_rejected(self):
        ar = self._ar_model()
        with tempfile.NamedTemporaryFile(suffix=".pt") as saved:
            torch.save({"model_state_dict": ar.state_dict()}, saved.name)
            loaded = UniformDiffusionModel.from_ar_checkpoint(saved.name, self.config).eval()
        torch.testing.assert_close(loaded.token_emb.weight, ar.token_emb.weight)
        bad = dict(ar.state_dict())
        bad.pop("layers.0.attn.q_proj.weight")
        with self.assertRaisesRegex(ValueError, "missing="):
            self.model.initialize_from_ar(bad)
        bad = dict(ar.state_dict())
        bad["lm_head.weight"] = bad["lm_head.weight"][:3]
        with self.assertRaisesRegex(ValueError, "shape_mismatches"):
            self.model.initialize_from_ar(bad)

    def test_incremental_causal_encoding_matches_full_encoding(self):
        first = self.model.encode_context(self.prompt[:, :3], genre=self.genre, multi_hot=self.multi_hot)
        before = tuple((k.clone(), v.clone()) for k, v in first.cache.key_values)
        appended = self.model.encode_context(self.prompt[:, 3:], cache=first.cache)
        full = self._context()
        torch.testing.assert_close(appended.logits, full.logits[:, 3:], atol=2e-6, rtol=2e-5)
        for got, wanted in zip(appended.cache.key_values, full.cache.key_values):
            torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-5)
        for got, wanted in zip(first.cache.key_values, before):
            torch.testing.assert_close(got, wanted)

    def test_context_and_canvas_padding_match_individual_unpadded_examples(self):
        context_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        canvas_mask = torch.tensor([[1] * 8, [1] * 4 + [0] * 4], dtype=torch.bool)
        batch_context = self._context(attention_mask=context_mask)
        batch_logits = self.model.denoise(self.canvas, cache=batch_context.cache, attention_mask=canvas_mask)
        one_context = self.model.encode_context(self.prompt[1:2, :2], genre=self.genre[1:2], multi_hot=self.multi_hot[1:2])
        one_logits = self.model.denoise(self.canvas[1:2, :4], cache=one_context.cache)
        torch.testing.assert_close(batch_logits[1, :4], one_logits[0], atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.all(batch_logits[1, 4:] == 0))
        self.assertEqual(batch_context.cache.next_positions.tolist(), [7, 4])
        # Appending after padding holes must also use true, not padded, positions.
        batch_append = self.model.encode_context(self.canvas[:, :2], cache=batch_context.cache)
        one_append = self.model.encode_context(self.canvas[1:2, :2], cache=one_context.cache)
        torch.testing.assert_close(batch_append.logits[1], one_append.logits[0], atol=2e-6, rtol=2e-5)
        # Padding values, including sampled PAD IDs at valid positions, have
        # no implicit masking semantics; only the supplied masks control it.
        other_prompt = self.prompt.clone()
        other_prompt[1, 2:] = (other_prompt[1, 2:] + 1) % 32
        other = self.model.encode_context(other_prompt, genre=self.genre, multi_hot=self.multi_hot, attention_mask=context_mask)
        torch.testing.assert_close(
            batch_logits, self.model.denoise(self.canvas, cache=other.cache, attention_mask=canvas_mask)
        )

    def test_causal_fast_path_only_for_right_padding(self):
        import torch.nn.functional as functional
        for mask, expected_causal in (([1, 1, 1, 0, 0], True), ([1, 0, 1, 0, 1], False)):
            valid = torch.tensor([mask], dtype=torch.bool)
            tokens = self.prompt[:1]
            with patch("improvnet.model.uniform_diffusion_model.F.scaled_dot_product_attention",
                       wraps=functional.scaled_dot_product_attention) as attention:
                padded = self.model.encode_context(tokens, attention_mask=valid,
                                                  genre=self.genre[:1], multi_hot=self.multi_hot[:1])
                self.assertEqual(attention.call_args.kwargs["is_causal"], expected_causal)
                if expected_causal:
                    self.assertIsNone(attention.call_args.kwargs["attn_mask"])
            unpadded = self.model.encode_context(tokens[:, valid[0]], genre=self.genre[:1], multi_hot=self.multi_hot[:1])
            torch.testing.assert_close(padded.logits[:, valid[0]], unpadded.logits, atol=2e-6, rtol=2e-5)

    def test_context_is_causal_canvas_is_bidirectional_and_cache_is_read_only(self):
        original = self._context()
        changed = self.prompt.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 32
        alternate = self.model.encode_context(changed, genre=self.genre, multi_hot=self.multi_hot)
        torch.testing.assert_close(original.logits[:, :-1], alternate.logits[:, :-1])
        before = tuple((k.clone(), v.clone()) for k, v in original.cache.key_values)
        first = self.model.denoise(self.canvas, cache=original.cache)
        modified_canvas = self.canvas.clone()
        modified_canvas[:, -1] = (modified_canvas[:, -1] + 1) % 32
        second = self.model.denoise(modified_canvas, cache=original.cache)
        self.assertGreater((first[:, 0] - second[:, 0]).abs().max().item(), 1e-6)
        for got, wanted in zip(original.cache.key_values, before):
            torch.testing.assert_close(got, wanted)
        self.assertEqual(original.cache.attention_mask.shape[1], 7)

    def test_prefix_view_hides_current_and_future_clean_targets(self):
        full = self._context()
        limited = full.cache.prefix(torch.tensor([2, 2]))
        direct = self.model.encode_context(self.prompt[:, :2], genre=self.genre, multi_hot=self.multi_hot)
        torch.testing.assert_close(
            self.model.denoise(self.canvas, cache=limited),
            self.model.denoise(self.canvas, cache=direct.cache), atol=2e-6, rtol=2e-5,
        )
        self.assertEqual(limited.next_positions.tolist(), [4, 4])
        with self.assertRaises(ValueError):
            full.cache.prefix(torch.tensor([6, 2]))

    def test_promptless_genre_instruments_and_optional_elasticity(self):
        empty = self.prompt[:, :0]
        context = self.model.encode_context(empty, genre=self.genre, multi_hot=self.multi_hot)
        self.assertEqual(context.logits.shape, (2, 0, 32))
        self.assertEqual(context.cache.next_positions.tolist(), [2, 2])
        elastic = self.model.encode_context(empty, genre=self.genre, multi_hot=self.multi_hot, elasticity=torch.tensor([0., 0.15]))
        self.assertEqual(elastic.cache.control_tokens, 3)
        logits = self.model.denoise(self.canvas, cache=elastic.cache)
        self.assertTrue(torch.isfinite(logits).all())
        other_genre = self.model.encode_context(empty, genre=(self.genre + 1) % 4, multi_hot=self.multi_hot)
        self.assertFalse(torch.allclose(self.model.denoise(self.canvas, cache=context.cache),
                                        self.model.denoise(self.canvas, cache=other_genre.cache)))
        with self.assertRaisesRegex(ValueError, "already cached"):
            self.model.encode_context(self.prompt, cache=context.cache, genre=self.genre)

    def test_soft_self_conditioning_dropout_and_gradient_detachment(self):
        context = self._context()
        previous = torch.randn(2, 8, 32, requires_grad=True)
        base = self.model.denoise(self.canvas, cache=context.cache)
        conditioned = self.model.denoise(
            self.canvas, cache=context.cache, self_conditioning_logits=previous,
            self_conditioning_mask=torch.tensor([True, False]),
        )
        torch.testing.assert_close(conditioned[1], base[1])
        self.assertGreater((conditioned[0] - base[0]).abs().max().item(), 1e-6)
        context.cache.key_values[0][0].retain_grad()
        conditioned.square().mean().backward()
        self.assertIsNone(previous.grad)
        for parameter in (self.model.layers[0].attn.q_proj.weight,
                          self.model.self_conditioning.ffn.w1.weight,
                          self.model.genre_emb.weight, self.model.multihot_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNotNone(context.cache.key_values[0][0].grad)
        self.assertIsNone(context.cache.detach().key_values[0][0].grad_fn)

    def test_combined_forward_checkpointing_matches_gradients_without_it(self):
        plain = self.model.train()
        checked = UniformDiffusionModel(replace(self.config, gradient_checkpointing=True)).train()
        checked.load_state_dict(plain.state_dict())
        outputs = []
        for model in (plain, checked):
            output = model(
                self.prompt, self.canvas, genre=self.genre, multi_hot=self.multi_hot,
                elasticity=torch.tensor([0.0, 0.15]), return_context_logits=True,
            )
            (output.logits.square().mean() + output.context.logits.square().mean()).backward()
            outputs.append(output)
        torch.testing.assert_close(outputs[0].logits, outputs[1].logits)
        for (name, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, msg=name)
        # One backbone in state_dict: no separately registered encoder/decoder copies.
        self.assertFalse(any(name.startswith(("encoder.", "decoder.")) for name in plain.state_dict()))


if __name__ == "__main__":
    unittest.main()
