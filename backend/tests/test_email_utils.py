import pytest

from app.services.email_utils import domain_matches, extract_email_tokens
from app.services.intent_filter import evaluate_predicate_row


@pytest.mark.parametrize(
    ("value", "addresses"),
    [
        (" ABC@GMAIL.COM ", ["ABC@gmail.com"]),
        ("a@gmail.com, b@yahoo.com", ["a@gmail.com", "b@yahoo.com"]),
        ("a@gmail.com; b@yahoo.com", ["a@gmail.com", "b@yahoo.com"]),
        ("a@gmail.com|b@yahoo.com", ["a@gmail.com", "b@yahoo.com"]),
        ("a@gmail.com\nb@yahoo.com", ["a@gmail.com", "b@yahoo.com"]),
        ("Alex Example <alex@gmail.com>", ["alex@gmail.com"]),
        ("not an email", []),
        ("broken@localhost", []),
        (None, []),
    ],
)
def test_extract_email_tokens(value, addresses):
    assert [item["address"] for item in extract_email_tokens(value)] == addresses


@pytest.mark.parametrize(
    ("domain", "is_cornell"),
    [
        ("cornell.edu", True),
        ("CORNELL.EDU", True),
        ("alumni.cornell.edu", True),
        ("cs.cornell.edu", True),
        ("cornelltech.io", False),
        ("cornell.edu.com", False),
        ("notcornell.edu", False),
        ("gmail.com", False),
    ],
)
def test_domain_matching_uses_dns_label_boundaries(domain, is_cornell):
    assert domain_matches(domain, "cornell.edu", include_subdomains=True) is is_cornell


def test_mixed_cornell_and_external_addresses_remain_distinct():
    tokens = extract_email_tokens("abc@cornell.edu; abc@gmail.com")
    assert [item["domain"] for item in tokens] == ["cornell.edu", "gmail.com"]


@pytest.mark.parametrize(
    ("value", "matches_external"),
    [
        ("abc123@cornell.edu", False),
        ("ABC123@CORNELL.EDU", False),
        ("abc@alumni.cornell.edu", False),
        ("abc@cs.cornell.edu", False),
        ("abc@gmail.com", True),
        ("abc@yahoo.com", True),
        ("cornellstudent@gmail.com", True),
        ("abc@cornelltech.io", True),
        ("abc@cornell.edu.com", True),
        ("abc@notcornell.edu", True),
        ("abc@cornell.edu; abc@gmail.com", True),
        ("abc@cornell.edu, abc@alumni.cornell.edu", False),
        ("", False),
        ("   ", False),
        (None, False),
        ("not an email", False),
        ("abc@cornell.edu ", False),
        ("ABC@GMAIL.COM", True),
    ],
)
def test_non_cornell_predicate_exact_edge_semantics(value, matches_external):
    predicate = {
        "columns": ["Email"],
        "operator": "email_domain_not_in",
        "values": ["cornell.edu"],
        "include_subdomains": True,
        "quantifier": "any",
        "require_valid_value": True,
    }
    assert evaluate_predicate_row({"Email": value}, predicate) is matches_external
