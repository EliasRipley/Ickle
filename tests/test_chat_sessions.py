import tempfile
import unittest

from src.chat_sessions import ChatSessions


class ChatSessionEpistemicsTests(unittest.TestCase):
    def test_answer_map_and_confidence_survive_session_reload(self):
        with tempfile.TemporaryDirectory() as td:
            sessions = ChatSessions(td)
            session = sessions.create_session("test")
            passport = {"version": 1, "claims": [{"claim_id": "c1", "text": "A candidate claim."}]}
            sessions.add_message(
                session["id"],
                "assistant",
                "An answer.",
                model="models/test.pt",
                epistemics=passport,
                low_confidence=True,
            )
            loaded = sessions.get_session(session["id"])
            message = loaded["messages"][0]
            self.assertEqual(message["epistemics"], passport)
            self.assertTrue(message["lowConfidence"])


if __name__ == "__main__":
    unittest.main()
