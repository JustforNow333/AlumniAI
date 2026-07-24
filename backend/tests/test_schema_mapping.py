import json
from io import BytesIO

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service, dataset_store
from app.services.canonical_schema import (
    CANONICAL_FIELD_REGISTRY,
    CANONICAL_FIELDS,
    validate_canonical_registry,
)
from app.services.column_resolver import (
    resolve_all_semantic_columns,
    resolve_canonical_column,
    resolve_person_columns,
)
from app.services.schema_inference import infer_schema_profile
from app.services.schema_profile import (
    SchemaProfileValidationError,
    validate_schema_update,
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    application = create_app()
    application.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
        HISTORY_REGISTRY_PATH=str(tmp_path / "data" / "history.json"),
    )
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def upload(client, df, filename="schema.csv"):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(df.to_csv(index=False).encode()), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def mapping_payload(**fields):
    return {
        "status": "confirmed",
        "mappings": {
            key: {"source_columns": value if isinstance(value, list) else [value]}
            for key, value in fields.items()
        },
        "ignored_columns": [],
    }


def test_canonical_registry_is_valid_unique_and_complete():
    assert validate_canonical_registry() is True
    keys = [field["key"] for field in CANONICAL_FIELDS]
    assert len(keys) == len(set(keys))
    assert {"constituent_id", "first_name", "employer", "email", "lifetime_giving", "do_not_contact"} <= set(keys)
    assert CANONICAL_FIELD_REGISTRY["email"]["cardinality"] == "multiple"
    assert CANONICAL_FIELD_REGISTRY["phone"]["cardinality"] == "multiple"


def test_familiar_and_compact_aliases_still_resolve():
    df = pd.DataFrame(
        columns=["First Name", "LAST_NAME", "LinkedInURL", "grad-yr", "Employer"]
    )
    assert resolve_canonical_column(df, "first_name") == "First Name"
    assert resolve_canonical_column(df, "last_name") == "LAST_NAME"
    assert resolve_canonical_column(df, "linkedin_url") == "LinkedInURL"
    assert resolve_canonical_column(df, "grad_year") == "grad-yr"
    assert resolve_canonical_column(df, "employer") == "Employer"


@pytest.mark.parametrize(
    ("header", "canonical"),
    [
        ("FIRST_NM", "first_name"),
        ("LAST_NM", "last_name"),
        ("BUSINESS_NAME_1", "employer"),
        ("PRIMARY_BUSINESS_POSITION", "occupation"),
        ("CONSTITUENT_PREFERRED_EMAIL", "email"),
        ("HOME_CITY_1", "city"),
        ("LT_GIFT_AMT", "lifetime_giving"),
        ("LAST_CONTACT_DT", "last_contact_date"),
    ],
)
def test_institutional_header_heuristics(header, canonical):
    values = ["Ada", "Grace"]
    if canonical == "email":
        values = ["ada@example.com", "grace@example.com"]
    elif canonical == "lifetime_giving":
        values = [100, 250]
    elif canonical == "last_contact_date":
        values = ["2025-01-01", "2026-02-02"]
    profile = infer_schema_profile(pd.DataFrame({header: values}))
    assert profile["mappings"][canonical]["source_columns"] == [header]
    assert profile["mappings"][canonical]["method"] in {
        "header_heuristic",
        "combined_inference",
    }


def test_sample_inference_email_linkedin_and_grad_year():
    email = infer_schema_profile(
        pd.DataFrame({"PREF_EML": ["a@example.com", "b@example.org", "c@test.net"]})
    )
    linkedin = infer_schema_profile(
        pd.DataFrame(
            {
                "PROFILE_LINK": [
                    "https://linkedin.com/in/a",
                    "linkedin.com/in/b",
                    "https://www.linkedin.com/in/c",
                ]
            }
        )
    )
    grad = infer_schema_profile(
        pd.DataFrame({"CLASS_CD": [2018, 2020, 2022]})
    )
    assert email["mappings"]["email"]["source_columns"] == ["PREF_EML"]
    assert linkedin["mappings"]["linkedin_url"]["source_columns"] == ["PROFILE_LINK"]
    assert grad["mappings"]["grad_year"]["source_columns"] == ["CLASS_CD"]


def test_notes_containing_emails_are_not_inferred_as_email():
    profile = infer_schema_profile(
        pd.DataFrame(
            {
                "CONTACT_NOTES": [
                    "Try a@example.com",
                    "Old address b@example.com",
                    "Assistant c@example.com",
                ]
            }
        )
    )
    assert "email" not in profile["mappings"]
    assert "CONTACT_NOTES" in profile["unmapped_columns"]


def test_multi_email_inference_and_resolution_preserve_all_columns():
    df = pd.DataFrame(
        {
            "Email 1": ["a@example.com", "b@example.com"],
            "Alternate Email": ["a@work.com", "b@work.com"],
        }
    )
    profile = infer_schema_profile(df)
    assert profile["mappings"]["email"]["source_columns"] == [
        "Email 1",
        "Alternate Email",
    ]
    context = {
        "columns": [{"name": column, "sample_values": df[column].tolist()} for column in df.columns],
        "schema_mapping": {
            "status": "confirmed",
            "canonical_to_source": {"email": list(df.columns)},
        },
    }
    assert resolve_all_semantic_columns("email", context, question="non-Cornell email") == list(df.columns)


def test_single_cardinality_conflict_is_visible_not_silently_selected():
    profile = infer_schema_profile(
        pd.DataFrame({"Employer": ["Acme"], "Company": ["Beta"]})
    )
    assert "employer" not in profile["mappings"]
    assert {
        "type": "single_field_multiple_sources",
        "canonical_field": "employer",
        "source_columns": ["Employer", "Company"],
    } in profile["conflicts"]


def test_schema_validation_rejects_invalid_fields_sources_and_duplicates():
    columns = ["A", "B"]
    with pytest.raises(SchemaProfileValidationError, match="Unknown canonical"):
        validate_schema_update(mapping_payload(invented_field="A"), columns)
    with pytest.raises(SchemaProfileValidationError, match="does not exist"):
        validate_schema_update(mapping_payload(employer="Missing"), columns)
    with pytest.raises(SchemaProfileValidationError, match="only one"):
        validate_schema_update(mapping_payload(employer=["A", "B"]), columns)
    with pytest.raises(SchemaProfileValidationError, match="assigned to both"):
        validate_schema_update(mapping_payload(employer="A", occupation="A"), columns)


def test_confirmed_mapping_overrides_misleading_aliases_everywhere():
    df = pd.DataFrame(
        {
            "Employer": ["Misleading Alias"],
            "BUS_NM": ["Spotify"],
            "BUS_POS": ["Director"],
        }
    )
    confirmed = validate_schema_update(
        mapping_payload(employer="BUS_NM", occupation="BUS_POS"),
        df.columns,
    )
    df.attrs["schema_profile"] = confirmed
    assert resolve_canonical_column(df, "employer") == "BUS_NM"
    assert resolve_person_columns(df)["occupation"] == "BUS_POS"


def test_reinference_preserves_confirmed_by_default_and_reset_discards_it():
    df = pd.DataFrame({"Employer": ["Alias Co"], "BUS_NM": ["Spotify"]})
    confirmed = validate_schema_update(mapping_payload(employer="BUS_NM"), df.columns)
    preserved = infer_schema_profile(df, existing_profile=confirmed)
    reset = infer_schema_profile(df, existing_profile=confirmed, reset_confirmed=True)
    assert preserved["mappings"]["employer"]["source_columns"] == ["BUS_NM"]
    assert preserved["mappings"]["employer"]["user_confirmed"] is True
    assert reset["mappings"]["employer"]["source_columns"] == ["Employer"]
    assert reset["mappings"]["employer"]["user_confirmed"] is False


def test_offline_and_malformed_or_invented_model_output_fall_back_safely():
    df = pd.DataFrame({"OPAQUE": ["one", "two", "three"]})
    offline = infer_schema_profile(df, use_model=True, ai_client=None)
    assert offline["mappings"] == {}

    class Responses:
        def __init__(self, text):
            self.text = text

        def create(self, **_kwargs):
            return type("Response", (), {"output_text": self.text})()

    class Client:
        def __init__(self, text):
            self.responses = Responses(text)

    malformed = infer_schema_profile(df, use_model=True, ai_client=Client("not json"))
    invented = infer_schema_profile(
        df,
        use_model=True,
        ai_client=Client(
            json.dumps(
                {
                    "suggestions": [
                        {
                            "source_column": "MADE_UP",
                            "canonical_field": "employer",
                            "confidence": 1,
                        },
                        {
                            "source_column": "OPAQUE",
                            "canonical_field": "made_up_field",
                            "confidence": 1,
                        },
                    ]
                }
            )
        ),
    )
    assert malformed["mappings"] == {}
    assert malformed["warnings"]
    assert invented["mappings"] == {}


def test_upload_get_put_list_and_restart_persist_schema(client, app):
    uploaded = upload(
        client,
        pd.DataFrame(
            {
                "FIRST_NM": ["Ada", "Grace"],
                "BUSINESS_NAME_1": ["Google", "Navy"],
                "CUSTOM": ["x", "y"],
            }
        ),
    )
    dataset_id = uploaded["dataset_id"]
    assert uploaded["metadata"]["schema_profile"]["status"] == "unreviewed"

    response = client.get(f"/api/datasets/{dataset_id}/schema")
    assert response.status_code == 200
    schema = response.get_json()
    assert schema["version"] == "schema_mapping_v1"
    assert schema["canonical_fields"]
    assert {column["name"] for column in schema["source_columns"]} == {
        "FIRST_NM",
        "BUSINESS_NAME_1",
        "CUSTOM",
    }

    saved = client.put(
        f"/api/datasets/{dataset_id}/schema",
        json={
            **mapping_payload(first_name="FIRST_NM", employer="BUSINESS_NAME_1"),
            "ignored_columns": ["CUSTOM"],
        },
    )
    assert saved.status_code == 200
    assert saved.get_json()["status"] == "confirmed"
    listed = client.get("/api/datasets").get_json()["datasets"][0]
    assert listed["schema_status"] == "confirmed"
    assert listed["schema_mapped_count"] == 2
    assert listed["schema_unmapped_count"] == 0

    restarted = create_app()
    restarted.config.update(
        TESTING=True,
        UPLOAD_FOLDER=app.config["UPLOAD_FOLDER"],
        DATA_FOLDER=app.config["DATA_FOLDER"],
        DATASET_REGISTRY_PATH=app.config["DATASET_REGISTRY_PATH"],
    )
    persisted = restarted.test_client().get(f"/api/datasets/{dataset_id}/schema")
    assert persisted.status_code == 200
    assert persisted.get_json()["mappings"]["employer"]["source_columns"] == [
        "BUSINESS_NAME_1"
    ]


def test_older_registry_lazily_receives_profile_and_listing_does_not_load_frames(client, app, monkeypatch):
    uploaded = upload(client, pd.DataFrame({"First Name": ["Ada"]}), "older.csv")
    dataset_id = uploaded["dataset_id"]
    with app.app_context():
        registry = dataset_store.load_dataset_registry()
        registry[dataset_id].pop("schema_profile")
        registry[dataset_id].pop("columns")
        dataset_store.save_dataset_registry(registry)

    original_reader = dataset_store.read_dataframe_from_path
    monkeypatch.setattr(
        dataset_store,
        "read_dataframe_from_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("listing loaded dataframe")),
    )
    listed = client.get("/api/datasets")
    assert listed.status_code == 200
    assert listed.get_json()["datasets"][0]["schema_status"] == "not_analyzed"
    monkeypatch.setattr(dataset_store, "read_dataframe_from_path", original_reader)

    lazy = client.get(f"/api/datasets/{dataset_id}/schema")
    assert lazy.status_code == 200
    with app.app_context():
        assert dataset_store.load_dataset_registry()[dataset_id]["schema_profile"]


def test_schema_errors_missing_file_rename_delete_and_atomic_failure(client, app, monkeypatch):
    uploaded = upload(client, pd.DataFrame({"A": ["Ada"]}), "lifecycle.csv")
    dataset_id = uploaded["dataset_id"]
    assert client.get("/api/datasets/missing/schema").status_code == 404

    renamed = client.patch(
        f"/api/datasets/{dataset_id}", json={"display_name": "Renamed"}
    )
    assert renamed.status_code == 200
    assert renamed.get_json()["schema_version"] == "schema_mapping_v1"

    with app.app_context():
        before = dataset_store.load_dataset_registry()
    monkeypatch.setattr(dataset_store.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    failed = client.put(
        f"/api/datasets/{dataset_id}/schema",
        json=mapping_payload(first_name="A"),
    )
    assert failed.status_code == 500
    registry_text = open(app.config["DATASET_REGISTRY_PATH"], encoding="utf-8").read()
    assert json.loads(registry_text) == before
    monkeypatch.undo()

    with app.app_context():
        stored = dataset_store.load_dataset_registry()[dataset_id]["stored_filename"]
        (dataset_store.get_storage_paths()["upload_folder"] / stored).unlink()
    assert client.get(f"/api/datasets/{dataset_id}/schema").status_code == 404
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200
    with app.app_context():
        assert dataset_id not in dataset_store.load_dataset_registry()


def test_confirmed_opaque_schema_changes_real_query_execution(client):
    df = pd.DataFrame(
        {
            "F_NM": ["Neil", "Ava", "Mia"],
            "L_NM": ["Wusu", "Lee", "Stone"],
            "BUS_NM": ["Spotify", "Google", "School"],
            "BUS_POS": ["Director Premium Strategy", "Software Engineer", "Teacher"],
            "PREF_EML": ["neil@example.com", "ava@cornell.edu", "mia@yahoo.com"],
            "ALT_EML": ["", "ava@gmail.com", ""],
            "CLASS_CD": [2018, 2022, 2022],
            # Deliberately misleading familiar aliases must lose to the profile.
            "Employer": ["Wrong Co", "Wrong Co", "Spotify"],
            "Occupation": ["Teacher", "Teacher", "Teacher"],
        }
    )
    dataset_id = upload(client, df, "opaque.csv")["dataset_id"]
    saved = client.put(
        f"/api/datasets/{dataset_id}/schema",
        json=mapping_payload(
            first_name="F_NM",
            last_name="L_NM",
            employer="BUS_NM",
            occupation="BUS_POS",
            email=["PREF_EML", "ALT_EML"],
            grad_year="CLASS_CD",
        ),
    )
    assert saved.status_code == 200

    employer = client.post(
        "/api/ask",
        json={"dataset_id": dataset_id, "question": "Who works at Spotify?"},
    ).get_json()
    occupation = client.post(
        "/api/ask",
        json={"dataset_id": dataset_id, "question": "Which alumni are software engineers?"},
    ).get_json()
    email = client.post(
        "/api/ask",
        json={"dataset_id": dataset_id, "question": "Show alumni with a non-Cornell email."},
    ).get_json()
    grad = client.post(
        "/api/ask",
        json={"dataset_id": dataset_id, "question": "Show alumni who graduated in 2022."},
    ).get_json()

    assert {row["First Name"] for row in employer["result"]["rows"]} == {"Neil"}
    assert {row["First Name"] for row in occupation["result"]["rows"]} == {"Ava"}
    assert {row["First Name"] for row in email["result"]["rows"]} == {"Neil", "Ava", "Mia"}
    assert {row["First Name"] for row in grad["result"]["rows"]} == {"Ava", "Mia"}
    for result in [employer["result"], occupation["result"], email["result"], grad["result"]]:
        assert "F_NM" not in result.get("columns", [])
        assert "BUS_NM" not in result.get("columns", [])

    # Changing the mapping changes later execution without touching the CSV.
    changed = client.put(
        f"/api/datasets/{dataset_id}/schema",
        json=mapping_payload(
            first_name="F_NM",
            last_name="L_NM",
            employer="Employer",
            occupation="BUS_POS",
            email=["PREF_EML", "ALT_EML"],
            grad_year="CLASS_CD",
        ),
    )
    assert changed.status_code == 200
    later = client.post(
        "/api/ask",
        json={"dataset_id": dataset_id, "question": "Who works at Spotify?"},
    ).get_json()
    assert {row["First Name"] for row in later["result"]["rows"]} == {"Mia"}
