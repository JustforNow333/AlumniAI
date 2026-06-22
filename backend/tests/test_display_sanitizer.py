from app.services.display_sanitizer import (
    question_requests_major,
    sanitize_display_rows,
    sanitize_response_payload,
)


def test_sanitizer_removes_internal_and_eval_columns_from_list_rows():
    columns = [
        "First Name",
        "Last Name",
        "Employer",
        "Title",
        "LinkedIn URL",
        "MATCH REASON",
        "match_reason",
        "Match Reason",
        "expected_industry",
        "expected_score",
        "eval_case_id",
        "classifier_reason",
        "confidence",
        "_temporary_rank",
    ]
    rows = [
        [
            "Ada",
            "Lovelace",
            "Analytical Engines",
            "Engineer",
            "linkedin.example/ada",
            "matched employer",
            "matched title",
            "matched major",
            "Tech",
            0.95,
            "case-1",
            "internal",
            0.99,
            1,
        ]
    ]

    safe_columns, safe_rows = sanitize_display_rows(columns, rows, "Show me alumni in tech")

    assert safe_columns == ["First Name", "Last Name", "Employer", "Title", "LinkedIn URL"]
    assert safe_rows == [
        ["Ada", "Lovelace", "Analytical Engines", "Engineer", "linkedin.example/ada"]
    ]


def test_sanitizer_removes_internal_columns_from_mapping_rows():
    columns = ["First Name", "Employer", "MATCH REASON", "expected_industry"]
    rows = [
        {
            "First Name": "Grace",
            "Employer": "Navy",
            "MATCH REASON": "matched title",
            "expected_industry": "Government",
            "extra": "not declared",
        }
    ]

    safe_columns, safe_rows = sanitize_display_rows(columns, rows, "Show alumni")

    assert safe_columns == ["First Name", "Employer"]
    assert safe_rows == [{"First Name": "Grace", "Employer": "Navy"}]


def test_major_is_hidden_unless_explicitly_requested():
    columns = ["First Name", "Employer", "Major", "LinkedIn URL"]
    rows = [["Ada", "Google", "Mathematics", "linkedin.example/ada"]]

    default_columns, default_rows = sanitize_display_rows(
        columns,
        rows,
        "Show me alumni at major tech companies",
    )
    requested_columns, requested_rows = sanitize_display_rows(
        columns,
        rows,
        "Show me alumni in tech and include their majors",
    )

    assert question_requests_major("Show me alumni at major tech companies") is False
    assert question_requests_major("What majors did tech alumni study?") is True
    assert default_columns == ["First Name", "Employer", "LinkedIn URL"]
    assert default_rows == [["Ada", "Google", "linkedin.example/ada"]]
    assert requested_columns == columns
    assert requested_rows == rows


def test_optional_user_fields_are_preserved_only_when_requested():
    columns = [
        "First Name",
        "Employer",
        "Notes",
        "Email",
        "Graduation Year",
        "Location",
    ]
    rows = [["Ada", "Google", "Board member", "ada@example.com", 2020, "New York"]]

    hidden_columns, _ = sanitize_display_rows(columns, rows, "Show me alumni at Google")
    visible_columns, visible_rows = sanitize_display_rows(
        columns,
        rows,
        "Show alumni at Google with notes, emails, graduation year, and location",
    )

    assert hidden_columns == ["First Name", "Employer"]
    assert visible_columns == columns
    assert visible_rows == rows


def test_sanitizer_handles_empty_inputs_and_preserves_row_count():
    assert sanitize_display_rows([], [], "Show alumni") == ([], [])

    columns = ["First Name", "MATCH REASON"]
    rows = [["Ada", "one"], ["Grace", "two"], ["Katherine", "three"]]
    safe_columns, safe_rows = sanitize_display_rows(columns, rows, "Show alumni")

    assert safe_columns == ["First Name"]
    assert len(safe_rows) == len(rows)


def test_response_payload_sanitizes_results_answer_blocks_and_debug_keys():
    payload = {
        "question": "Show me alumni in tech",
        "answer": {
            "summary": "Two matches",
            "blocks": [
                {
                    "type": "table",
                    "columns": ["First Name", "Major", "MATCH REASON"],
                    "rows": [["Ada", "Math", "matched title"]],
                }
            ],
            "followups": [],
        },
        "result": {
            "columns": ["First Name", "Major", "MATCH REASON"],
            "rows": [["Ada", "Math", "matched title"]],
            "display_columns": ["First Name", "Major", "MATCH REASON"],
            "debug": {"rows": [{"classification": "direct_match"}]},
        },
        "operation_results": [
            {
                "columns": ["First Name", "expected_industry", "confidence"],
                "rows": [["Ada", "Tech", 0.99]],
                "debug": {"rows": []},
            }
        ],
    }

    sanitized = sanitize_response_payload(payload)

    assert sanitized["answer"]["blocks"][0]["columns"] == ["First Name"]
    assert sanitized["answer"]["blocks"][0]["rows"] == [["Ada"]]
    assert sanitized["result"]["columns"] == ["First Name"]
    assert sanitized["result"]["rows"] == [["Ada"]]
    assert "debug" not in sanitized["result"]
    assert sanitized["operation_results"][0]["columns"] == ["First Name"]
    assert sanitized["operation_results"][0]["rows"] == [["Ada"]]
