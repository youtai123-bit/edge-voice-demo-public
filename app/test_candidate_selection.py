#!/usr/bin/env python3

import unittest

from candidate_selection import CandidateEvidence, normalize_text, select_candidate


def item(name: str, text: str, acoustic: float = 1.0, total_ms: int = 100):
    return CandidateEvidence(
        name=name,
        label=name,
        text=text,
        process_ms=total_ms // 2,
        decode_ms=total_ms - total_ms // 2,
        acoustic_score=acoustic,
        rms_dbfs=-20.0,
        clipping_ratio=0.0,
    )


class CandidateSelectionTest(unittest.TestCase):
    def test_normalize_chinese_punctuation_and_full_width(self):
        self.assertEqual(normalize_text(" 你好，ＡＢＣ １２！"), "你好ABC12")

    def test_normalize_preserves_japanese_and_korean(self):
        self.assertEqual(normalize_text("そうですね。"), "そうですね")
        self.assertEqual(normalize_text("カタカナ・テスト"), "カタカナテスト")
        self.assertEqual(normalize_text("안녕하세요!"), "안녕하세요")

    def test_japanese_literal_majority_is_not_discarded(self):
        decision = select_candidate(
            [item("raw", "そう"), item("gtcrn", "そう"), item("fastenhancer", "そう")]
        )
        self.assertEqual(decision.text, "そう")
        self.assertEqual(decision.reliability, "high")

    def test_literal_majority_wins(self):
        decision = select_candidate(
            [
                item("raw", "打开会议记录", 0.8, 80),
                item("gtcrn", "打开会议记录", 0.9, 130),
                item("fastenhancer", "打开回忆记录", 1.0, 70),
            ]
        )
        self.assertEqual(decision.text, "打开会议记录")
        self.assertEqual(decision.selected_name, "gtcrn")
        self.assertIn("多数", decision.reason)

    def test_all_equal_is_high_reliability(self):
        decision = select_candidate(
            [item("raw", "测试"), item("gtcrn", "测试"), item("fastenhancer", "测试")]
        )
        self.assertEqual(decision.reliability, "high")
        self.assertEqual(decision.consensus_score, 1.0)

    def test_empty_candidates_do_not_win(self):
        decision = select_candidate(
            [item("raw", ""), item("gtcrn", "树莓派"), item("fastenhancer", "")]
        )
        self.assertEqual(decision.selected_name, "gtcrn")
        self.assertEqual(decision.text, "树莓派")

    def test_all_empty_returns_none(self):
        decision = select_candidate(
            [item("raw", ""), item("gtcrn", ""), item("fastenhancer", "")]
        )
        self.assertEqual(decision.selected_name, "none")
        self.assertEqual(decision.text, "")

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate([item("raw", "甲"), item("raw", "乙")])


if __name__ == "__main__":
    unittest.main()
