import tempfile
import unittest
from pathlib import Path

from src.federated.codistill import (
    DEFAULT_TRUST,
    GENERAL_DOMAIN,
    PeerTrustStore,
    Probe,
    TeachingResponse,
    ask_swarm,
    classify_domain,
    default_probes,
    domain_of_probe,
    load_probes,
    run_codistillation_round,
    score_consensus,
    select_probes_for_round,
    write_distillation_corpus,
)
from src.federated.contribution_ledger import LedgerStore
from src.federated.inference_swarm import InferenceNode
from src.federated.keys import create_ed_identity
from src.federated.peer_discovery import PeerDiscovery


def _stub_generate_factory(text: str):
    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        return text
    return _generate


class ProbeLoadingTests(unittest.TestCase):
    def test_missing_file_falls_back_to_defaults(self):
        probes = load_probes("data/does_not_exist/probe_prompts.json")
        self.assertEqual(probes, default_probes())

    def test_loads_shipped_probe_file(self):
        probes = load_probes("data/codistill/probe_prompts.json")
        self.assertGreater(len(probes), 10)
        self.assertTrue(all(isinstance(p, Probe) and p.prompt for p in probes))

    def test_malformed_json_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            probes = load_probes(path)
            self.assertEqual(probes, default_probes())


class ConsensusScoringTests(unittest.TestCase):
    def test_single_response_gets_neutral_score(self):
        responses = [TeachingResponse(probe_id="p1", prompt="q", teacher_peer_id="a", response="the sky is blue")]
        score_consensus(responses)
        self.assertEqual(responses[0].consensus_score, 1.0)

    def test_agreeing_responses_score_higher_than_outlier(self):
        responses = [
            TeachingResponse(probe_id="p1", prompt="q", teacher_peer_id="a", response="weather changes day to day while climate is the long term average"),
            TeachingResponse(probe_id="p1", prompt="q", teacher_peer_id="b", response="climate is the long term average while weather changes day to day"),
            TeachingResponse(probe_id="p1", prompt="q", teacher_peer_id="c", response="bananas are a good source of potassium"),
        ]
        score_consensus(responses)
        self.assertGreater(responses[0].consensus_score, responses[2].consensus_score)
        self.assertGreater(responses[1].consensus_score, responses[2].consensus_score)


class PeerTrustStoreTests(unittest.TestCase):
    def test_unknown_peer_gets_default_trust(self):
        with tempfile.TemporaryDirectory() as d:
            store = PeerTrustStore(Path(d) / "trust.json")
            self.assertEqual(store.get("nobody"), DEFAULT_TRUST)

    def test_update_moves_trust_toward_observed_score(self):
        with tempfile.TemporaryDirectory() as d:
            store = PeerTrustStore(Path(d) / "trust.json")
            for _ in range(20):
                store.update("peer-good", 1.0)
                store.update("peer-bad", 0.0)
            self.assertGreater(store.get("peer-good"), 0.9)
            self.assertLess(store.get("peer-bad"), 0.1)

    def test_save_and_reload_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trust.json"
            store = PeerTrustStore(path)
            store.update("peer-x", 0.8)
            store.save()
            reloaded = PeerTrustStore(path)
            self.assertAlmostEqual(reloaded.get("peer-x"), store.get("peer-x"))

    def test_ranked_peer_ids_orders_by_trust_descending(self):
        with tempfile.TemporaryDirectory() as d:
            store = PeerTrustStore(Path(d) / "trust.json")
            store.trust = {"low": {"general": 0.1}, "high": {"general": 0.9}, "mid": {"general": 0.5}}
            self.assertEqual(store.ranked_peer_ids(), ["high", "mid", "low"])

    def test_domain_trust_is_independent_per_domain(self):
        with tempfile.TemporaryDirectory() as d:
            store = PeerTrustStore(Path(d) / "trust.json")
            for _ in range(20):
                store.update("peer-x", 1.0, domain="code")
                store.update("peer-x", 0.0, domain="creative")
            self.assertGreater(store.get("peer-x", "code"), 0.9)
            self.assertLess(store.get("peer-x", "creative"), 0.1)

    def test_unknown_domain_falls_back_to_overall(self):
        with tempfile.TemporaryDirectory() as d:
            store = PeerTrustStore(Path(d) / "trust.json")
            store.update("peer-y", 0.8, domain="code")
            self.assertAlmostEqual(store.get("peer-y", "math"), store.overall("peer-y"))

    def test_legacy_flat_store_migrates_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trust.json"
            path.write_text('{"peer-old": 0.7}', encoding="utf-8")
            store = PeerTrustStore(path)
            self.assertEqual(store.get("peer-old"), 0.7)


class DomainClassificationTests(unittest.TestCase):
    def test_code_keywords_classify_as_code(self):
        self.assertEqual(classify_domain("Write a Python function that sorts a list"), "code")

    def test_math_keywords_classify_as_math(self):
        self.assertEqual(classify_domain("Calculate 17% of 240"), "math")

    def test_unmatched_text_falls_back_to_general(self):
        self.assertEqual(classify_domain("xyz abc qqq"), GENERAL_DOMAIN)

    def test_domain_of_probe_uses_id_prefix(self):
        self.assertEqual(domain_of_probe("code-03"), "code")
        self.assertEqual(domain_of_probe("noprefix"), "noprefix")


class ProbeRotationTests(unittest.TestCase):
    def test_sample_size_zero_returns_all_probes(self):
        probes = default_probes()
        self.assertEqual(select_probes_for_round(probes, 0), probes)

    def test_sample_size_limits_count(self):
        probes = load_probes("data/codistill/probe_prompts.json")
        sampled = select_probes_for_round(probes, 5, seed=42)
        self.assertEqual(len(sampled), 5)

    def test_same_seed_is_deterministic(self):
        probes = load_probes("data/codistill/probe_prompts.json")
        a = select_probes_for_round(probes, 6, seed=99)
        b = select_probes_for_round(probes, 6, seed=99)
        self.assertEqual([p.probe_id for p in a], [p.probe_id for p in b])


class WriteDistillationCorpusTests(unittest.TestCase):
    def test_writes_dialog_pair_format(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "corpus.txt"
            rows = [{"user": "What is 2+2?", "assistant": "4", "source": "codistill:abc"}]
            write_distillation_corpus(out, rows)
            content = out.read_text(encoding="utf-8")
            self.assertIn("User: What is 2+2?", content)
            self.assertIn("Ickle: 4", content)


class EndToEndRoundTests(unittest.TestCase):
    """Runs a real in-process swarm of heterogeneous 'peers' (stand-ins for
    different-architecture nodes -- the transport doesn't care what's behind
    generate_fn) over real HTTP, the same pattern test_inference_swarm.py
    uses, to prove the full discover -> ask -> consensus-filter -> corpus
    pipeline works without a coordinator."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        self.peer_discovery = PeerDiscovery()
        self.nodes: list[InferenceNode] = []

    def tearDown(self):
        for node in self.nodes:
            node.stop()
        self.tmpdir.cleanup()

    def _start_teacher(self, label: str, response_text: str) -> InferenceNode:
        identity = create_ed_identity(label=label)
        ledger = LedgerStore(self.data_dir / f"{label}_ledger.json")
        node = InferenceNode(
            identity=identity,
            generate_fn=_stub_generate_factory(response_text),
            model_hash=f"model-{label}",  # deliberately different "architectures"
            data_dir=self.data_dir,
            peer_discovery=self.peer_discovery,
            ledger=ledger,
            host="127.0.0.1",
            port=0,
            external_host="127.0.0.1",
            capacity=2,
            label=label,
        )
        node.start()
        node.announce()
        self.nodes.append(node)
        return node

    def test_consensus_answer_wins_over_lone_outlier(self):
        self._start_teacher("nano", "the sky is blue because of rayleigh scattering of sunlight")
        self._start_teacher("laptop", "sunlight scatters and the blue wavelength scatters most, making the sky look blue")
        self._start_teacher("desktop", "bananas are yellow")  # outlier / low-effort answer

        learner_identity = create_ed_identity(label="learner")
        learner_ledger = LedgerStore(self.data_dir / "learner_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "trust.json")

        probes = [Probe(probe_id="sky-01", prompt="Why is the sky blue?")]
        report = run_codistillation_round(
            identity=learner_identity,
            peer_discovery=self.peer_discovery,
            ledger=learner_ledger,
            trust_store=trust_store,
            probes=probes,
            max_teachers_per_probe=5,
            min_teachers_per_probe=2,
            min_consensus=0.15,
        )

        self.assertEqual(report["peers_discovered"], 3)
        self.assertEqual(report["probes_taught"], 1)
        row = report["rows"][0]
        self.assertIn("scattering", row["assistant"].lower())
        self.assertNotIn("bananas", row["assistant"].lower())

    def test_too_few_teachers_skips_probe(self):
        self._start_teacher("solo", "an answer nobody can corroborate")

        learner_identity = create_ed_identity(label="learner2")
        learner_ledger = LedgerStore(self.data_dir / "learner2_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "trust2.json")

        probes = [Probe(probe_id="p1", prompt="Anything?")]
        report = run_codistillation_round(
            identity=learner_identity,
            peer_discovery=self.peer_discovery,
            ledger=learner_ledger,
            trust_store=trust_store,
            probes=probes,
            min_teachers_per_probe=2,
        )
        self.assertEqual(report["probes_taught"], 0)

    def test_ledger_credits_both_sides(self):
        teacher = self._start_teacher("teacher1", "a solid, corroborated answer about gravity")
        self._start_teacher("teacher2", "a solid corroborated answer about gravity and mass")

        learner_identity = create_ed_identity(label="learner3")
        learner_ledger = LedgerStore(self.data_dir / "learner3_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "trust3.json")

        run_codistillation_round(
            identity=learner_identity,
            peer_discovery=self.peer_discovery,
            ledger=learner_ledger,
            trust_store=trust_store,
            probes=[Probe(probe_id="grav-01", prompt="What is gravity?")],
            min_teachers_per_probe=2,
        )

        self.assertGreater(learner_ledger.ledger.peer_requests_consumed, 0)
        self.assertGreater(teacher.ledger.ledger.peer_requests_served, 0)

    def test_agreement_does_not_bootstrap_trust_after_round(self):
        self._start_teacher("t1", "consistent answer about photosynthesis using sunlight")
        self._start_teacher("t2", "consistent answer about photosynthesis using sunlight and water")

        learner_identity = create_ed_identity(label="learner4")
        learner_ledger = LedgerStore(self.data_dir / "learner4_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "trust4.json")

        run_codistillation_round(
            identity=learner_identity,
            peer_discovery=self.peer_discovery,
            ledger=learner_ledger,
            trust_store=trust_store,
            probes=[Probe(probe_id="photo-01", prompt="What is photosynthesis?")],
            min_teachers_per_probe=2,
        )
        self.assertEqual(trust_store.trust, {})


class AskSwarmLiveTests(unittest.TestCase):
    """ask_swarm() is the live, in-chat sibling of run_codistillation_round():
    same signed-offer transport and trust store, but a single ad-hoc prompt
    answered concurrently instead of a probe set answered for a future
    training corpus. Reuses the same in-process InferenceNode harness as
    EndToEndRoundTests above."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        self.peer_discovery = PeerDiscovery()
        self.nodes: list[InferenceNode] = []

    def tearDown(self):
        for node in self.nodes:
            node.stop()
        self.tmpdir.cleanup()

    def _start_teacher(self, label: str, response_text: str) -> InferenceNode:
        identity = create_ed_identity(label=label)
        ledger = LedgerStore(self.data_dir / f"{label}_ledger.json")
        node = InferenceNode(
            identity=identity,
            generate_fn=_stub_generate_factory(response_text),
            model_hash=f"model-{label}",
            data_dir=self.data_dir,
            peer_discovery=self.peer_discovery,
            ledger=ledger,
            host="127.0.0.1",
            port=0,
            external_host="127.0.0.1",
            capacity=2,
            label=label,
        )
        node.start()
        node.announce()
        self.nodes.append(node)
        return node

    def test_returns_representative_and_preserves_trust_for_human_review(self):
        self._start_teacher("peerA", "paris is the capital of france")
        self._start_teacher("peerB", "the capital of france is paris")
        self._start_teacher("peerC", "bananas are a good source of potassium and fiber")

        asker_identity = create_ed_identity(label="asker")
        asker_ledger = LedgerStore(self.data_dir / "asker_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "trust.json")

        result = ask_swarm(
            "What is the capital of France?",
            self_peer_id=asker_identity.peer_id,
            peer_discovery=self.peer_discovery,
            ledger=asker_ledger,
            trust_store=trust_store,
        )

        self.assertEqual(result["peers_asked"], 3)
        self.assertEqual(result["peers_answered"], 3)
        self.assertIsNotNone(result["best"])
        self.assertIn("paris", result["best"]["response"].lower())
        self.assertEqual(trust_store.trust, {})
        self.assertIn("deliberation", result)
        self.assertGreaterEqual(result["deliberation"]["summary"]["common_claims"], 1)

    def test_no_peers_returns_empty_result(self):
        asker_identity = create_ed_identity(label="asker-lonely")
        asker_ledger = LedgerStore(self.data_dir / "lonely_ledger.json")
        trust_store = PeerTrustStore(self.data_dir / "lonely_trust.json")

        result = ask_swarm(
            "Anybody there?",
            self_peer_id=asker_identity.peer_id,
            peer_discovery=self.peer_discovery,
            ledger=asker_ledger,
            trust_store=trust_store,
        )
        self.assertEqual(result["peers_asked"], 0)
        self.assertEqual(result["peers_answered"], 0)
        self.assertIsNone(result["best"])


if __name__ == "__main__":
    unittest.main()
