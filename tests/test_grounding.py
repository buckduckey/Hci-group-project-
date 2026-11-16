"""Offline tests for the grounding checker. These do not need Ollama."""
import unittest

from tests.grounding import check_answer, check_evidence

FORM = {
    "q1": "6", "q2": False, "q3": "Shopping", "q4": "Deliberate",
    "q5": True, "q6": False, "q7": False, "q8": False,
    "q9": "VPN", "q10": False, "q11": True,
}


def kinds(evidence, form=FORM):
    return [kind for kind, _ in check_evidence(evidence, form)]


class GroundedEvidence(unittest.TestCase):
    def test_correct_hours_pass(self):
        self.assertEqual(kinds("6 hours of daily digital immersion."), [])

    def test_decimal_hours_match_the_form(self):
        self.assertEqual(kinds("7.5 hours a day.", dict(FORM, q1="7.5")), [])

    def test_settings_quoted_as_they_are_pass(self):
        self.assertEqual(kinds("Late-night usage is False and device telemetry is tracked."), [])

    def test_true_and_false_spelled_out_pass(self):
        self.assertEqual(kinds("Cross-Device Geolocation Linked: True, Dark Patterns Present: False"), [])

    def test_selected_options_pass(self):
        self.assertEqual(kinds("Deliberate cadence on shopping sites while a VPN is used."), [])

    def test_denying_an_unselected_option_passes(self):
        self.assertEqual(kinds("No ad-blocker in use.", dict(FORM, q9="None")), [])

    def test_no_privacy_tooling_passes_when_none_is_selected(self):
        self.assertEqual(kinds("No privacy tooling is used.", dict(FORM, q9="None")), [])


class UngroundedEvidence(unittest.TestCase):
    def test_issue_example_nine_hours_is_an_example_leak(self):
        self.assertEqual(kinds("9 hours of daily digital immersion."), ["example_leak"])

    def test_other_wrong_number(self):
        self.assertEqual(kinds("12 hours of daily use."), ["wrong_number"])

    def test_issue_example_geolocation_contradiction(self):
        self.assertEqual(kinds("No cross-device geolocation linked."), ["contradiction"])

    def test_setting_claimed_on_when_off(self):
        self.assertEqual(kinds("Late-night usage is True."), ["contradiction"])

    def test_setting_claimed_on_without_the_word_true(self):
        self.assertEqual(kinds("Dark patterns are present."), ["contradiction"])

    def test_unselected_option(self):
        self.assertEqual(kinds("Rapid interaction cadence."), ["wrong_value"])

    def test_privacy_tooling_denied_when_vpn_is_selected(self):
        self.assertEqual(kinds("No privacy tooling is used."), ["contradiction"])

    def test_evidence_that_cites_nothing(self):
        self.assertEqual(kinds("They seem like a careful person."), ["no_citation"])

    def test_each_clause_is_judged_separately(self):
        self.assertEqual(
            kinds("No dark patterns present and late-night usage is True."),
            ["contradiction"],
        )


class WholeAnswer(unittest.TestCase):
    def test_reports_which_item_was_wrong(self):
        answer = {
            "key_info": [
                {"info": "a", "evidence": "6 hours of daily use.", "confidence": 50},
                {"info": "b", "evidence": "9 hours of daily use.", "confidence": 50},
            ],
            "conclusion": "c",
        }
        found = check_answer(answer, FORM)
        self.assertEqual([(index, kind) for index, kind, _, _ in found], [(1, "example_leak")])


if __name__ == "__main__":
    unittest.main()
