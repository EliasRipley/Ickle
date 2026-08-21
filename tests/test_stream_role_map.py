import unittest

from src.train import apply_stream_role_map, _parse_stream_role_map


class StreamRoleMapTests(unittest.TestCase):
    """Regression coverage: streaming Anthropic/hh-rlhf's chosen/rejected
    fields (formatted 'Human: ...\\n\\nAssistant: ...') without converting
    to Ickle's own 'User:'/'Ickle:' turn markers meant build_loss_mask()
    never found its response_prefix, silently fell back to unmasked
    raw-text training, and the model could pick up literal 'Human:'-shaped
    fragments as ordinary content -- confirmed live as incoherent,
    'User'-token-leaking output from a real fine-tuning run."""

    def test_default_map_remaps_human_and_assistant(self):
        role_map = _parse_stream_role_map("Human=User,Assistant=Ickle")
        text = "\n\nHuman: hi\n\nAssistant: hello there"
        out = apply_stream_role_map(text, role_map)
        self.assertNotIn("Human:", out)
        self.assertNotIn("Assistant:", out)
        self.assertIn("User: hi", out)
        self.assertIn("Ickle: hello there", out)

    def test_matches_real_hh_rlhf_shape(self):
        role_map = _parse_stream_role_map("Human=User,Assistant=Ickle")
        sample = (
            "\n\nHuman: What are some cuss words in english?"
            "\n\nAssistant: Here's an incomplete list.\n\nAss, dick, bugger."
            "\n\nHuman: What about ones with racial connotations?"
            "\n\nAssistant: I won't help with that."
        )
        out = apply_stream_role_map(sample, role_map)
        self.assertEqual(out.count("User:"), 2)
        self.assertEqual(out.count("Ickle:"), 2)

    def test_empty_expression_yields_no_pairs_and_is_a_no_op(self):
        role_map = _parse_stream_role_map("")
        text = "\n\nHuman: hi\n\nAssistant: hello"
        self.assertEqual(role_map, [])
        self.assertEqual(apply_stream_role_map(text, role_map), text)

    def test_malformed_pairs_are_skipped_not_fatal(self):
        role_map = _parse_stream_role_map("Human,Assistant=Ickle,=Foo,Bar=")
        self.assertEqual(role_map, [("Assistant", "Ickle")])

    def test_plain_text_with_no_markers_is_unaffected(self):
        role_map = _parse_stream_role_map("Human=User,Assistant=Ickle")
        text = "Ordinary web text with no dialogue markers at all."
        self.assertEqual(apply_stream_role_map(text, role_map), text)


if __name__ == "__main__":
    unittest.main()
