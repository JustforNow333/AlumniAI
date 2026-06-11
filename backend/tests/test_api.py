import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service
from app.services.dataset_store import load_dataset_registry


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Customer": ["Acme", "Beta", "Core", "Delta", "Echo", "Faro"],
            "Category": ["A", "B", "A", "B", "C", "A"],
            "Revenue": [100.0, 250.0, 300.0, 150.0, 500.0, 50.0],
            "Orders": [1, 2, 3, 2, 5, 1],
            "Date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]
            ),
        }
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)

    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )

    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def dataframe_to_csv_upload(df, filename="sample.csv"):
    payload = df.to_csv(index=False).encode("utf-8")
    return BytesIO(payload), filename


def dataframe_to_xlsx_upload(df, filename="sample.xlsx"):
    stream = BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    stream.seek(0)
    return stream, filename


def upload_dataframe(client, df, filename="sample.xlsx"):
    if filename.lower().endswith(".csv"):
        file_obj, upload_name = dataframe_to_csv_upload(df, filename)
    else:
        file_obj, upload_name = dataframe_to_xlsx_upload(df, filename)

    response = client.post(
        "/api/upload",
        data={"file": (file_obj, upload_name)},
        content_type="multipart/form-data",
    )
    assert response.status_code in {200, 201}, response.get_data(as_text=True)
    data = response.get_json()
    assert data["dataset_id"]
    return data["dataset_id"], data


def ask(client, dataset_id, question):
    return client.post("/api/ask", json={"dataset_id": dataset_id, "question": question})


def assert_structured_answer(data, expected_text=None):
    answer = data["answer"]
    assert isinstance(answer, dict)
    assert isinstance(answer["summary"], str)
    assert isinstance(answer["blocks"], list)
    assert isinstance(answer["followups"], list)
    assert data["answer_text"] == answer["summary"]
    if expected_text:
        assert expected_text in answer["summary"].lower()
    for block in answer["blocks"]:
        assert block["type"] in {"markdown", "table", "metrics", "ranked_list"}
        assert "<script" not in json.dumps(block).lower()
    return answer


def result_rows_by_first_column(result):
    return {row[0]: row[1] for row in result["rows"]}


def result_row_map(result, first_column_value):
    for row in result["rows"]:
        if row and row[0] == first_column_value:
            return dict(zip(result["columns"], row))
    raise AssertionError(f"Missing result row for {first_column_value!r}")


def test_flask_app_imports_correctly(app):
    assert app.testing is True
    assert app.test_client() is not None


def test_health_route(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()["status"] == "ok"


def test_upload_valid_csv(client, sample_df):
    response = client.post(
        "/api/upload",
        data={"file": dataframe_to_csv_upload(sample_df, "customers.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code in {200, 201}
    data = response.get_json()
    assert data["dataset_id"]
    assert data["filename"] == "customers.csv"
    assert data["summary"]["rows"] == len(sample_df)
    assert data["summary"]["columns"] == len(sample_df.columns)
    assert data["summary"]["column_names"] == list(sample_df.columns)


def test_upload_valid_xlsx(client, sample_df):
    dataset_id, data = upload_dataframe(client, sample_df, "customers.xlsx")

    assert dataset_id
    assert data["filename"] == "customers.xlsx"
    assert data["summary"]["rows"] == len(sample_df)
    assert data["summary"]["columns"] == len(sample_df.columns)
    assert data["summary"]["column_names"] == list(sample_df.columns)


def test_upload_creates_dataset_metadata(client, app, sample_df):
    dataset_id, data = upload_dataframe(client, sample_df, "metadata.csv")
    registry_path = Path(app.config["DATASET_REGISTRY_PATH"])

    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert dataset_id in registry

    metadata = registry[dataset_id]
    stored_path = Path(app.config["UPLOAD_FOLDER"]) / metadata["stored_filename"]

    assert stored_path.exists()
    assert data["metadata"]["dataset_id"] == dataset_id
    assert metadata["original_filename"] == "metadata.csv"
    assert metadata["stored_filename"].startswith(f"{dataset_id}_")
    assert metadata["file_path"] == f"uploads/{metadata['stored_filename']}"
    assert metadata["file_type"] == "csv"
    assert metadata["row_count"] == len(sample_df)
    assert metadata["column_count"] == len(sample_df.columns)
    assert metadata["columns"] == list(sample_df.columns)


def test_upload_missing_file(client):
    response = client.post("/api/upload", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_unsupported_file_type(client):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b"not a spreadsheet"), "sample.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "unsupported" in response.get_json()["error"].lower()


def test_upload_empty_csv_with_headers(client):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b"Customer,Category,Revenue,Orders,Date\n"), "empty.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code in {200, 201, 400}
    data = response.get_json()
    if response.status_code == 400:
        assert "error" in data
    else:
        assert data["dataset_id"]
        assert data["summary"]["rows"] == 0
        assert data["summary"]["column_names"] == ["Customer", "Category", "Revenue", "Orders", "Date"]


def test_preview_uploaded_xlsx(client, sample_df):
    large_df = pd.concat([sample_df] * 3, ignore_index=True)
    dataset_id, _ = upload_dataframe(client, large_df, "large-preview.xlsx")

    response = client.get(f"/api/datasets/{dataset_id}/preview")

    assert response.status_code == 200
    data = response.get_json()
    assert data["dataset_id"] == dataset_id
    assert data["filename"] == "large-preview.xlsx"
    assert data["row_count"] == len(large_df)
    assert data["column_count"] == len(sample_df.columns)
    assert data["missing_count"] == 0
    assert data["columns"] == list(sample_df.columns)
    assert set(data["data_types"]) == set(sample_df.columns)
    assert data["missing_values"] == {column: 0 for column in sample_df.columns}
    assert len(data["rows"]) == 10
    assert data["column_names"] == list(sample_df.columns)
    assert len(data["preview"]) == 10
    assert len(data["preview"]) < len(large_df)


def test_dataset_survives_simulated_restart(client, app, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "restart.xlsx")

    with app.app_context():
        registry = load_dataset_registry()

    assert dataset_id in registry

    response = client.get(f"/api/datasets/{dataset_id}/preview")

    assert response.status_code == 200
    data = response.get_json()
    assert data["column_names"] == list(sample_df.columns)
    assert len(data["preview"]) == len(sample_df)


def test_preview_invalid_dataset_id(client):
    response = client.get("/api/datasets/fake-id/preview")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_summary_uploaded_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "summary.xlsx")

    response = client.get(f"/api/datasets/{dataset_id}/summary")

    assert response.status_code == 200
    data = response.get_json()
    assert data["rows"] == len(sample_df)
    assert data["columns"] == len(sample_df.columns)
    assert data["column_names"] == list(sample_df.columns)
    assert "missing_values" in data
    assert "numeric_summary" in data
    assert "categorical_summary" in data


def test_summary_invalid_dataset_id(client):
    response = client.get("/api/datasets/fake-id/summary")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_ask_without_dataset_id(client):
    response = client.post("/api/ask", json={"question": "Summarize this dataset."})

    assert response.status_code == 400
    assert "dataset_id" in response.get_json()["error"]


def test_ask_without_question(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "ask-no-question.xlsx")

    response = client.post("/api/ask", json={"dataset_id": dataset_id})

    assert response.status_code == 400
    assert "question" in response.get_json()["error"].lower()


def test_ask_with_invalid_dataset_id(client):
    response = client.post(
        "/api/ask",
        json={"dataset_id": "fake-id", "question": "Summarize this dataset."},
    )

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_ask_summary_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "ask-summary.xlsx")

    response = ask(client, dataset_id, "Summarize this dataset.")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "column_summary"
    assert data["result"]["status"] == "ok"
    assert data["result"]["metrics"]["total_rows"] == len(sample_df)
    assert data["result"]["metrics"]["columns_analyzed"] == len(sample_df.columns)
    answer = assert_structured_answer(data)
    assert any(block["type"] == "metrics" for block in answer["blocks"])


def test_ask_missing_values_question_on_xlsx(client, sample_df):
    df = sample_df.copy()
    df.loc[1, "Revenue"] = None
    dataset_id, _ = upload_dataframe(client, df, "missing-values.xlsx")

    response = ask(client, dataset_id, "Are there any missing values?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "missing_values"
    assert result_rows_by_first_column(data["result"])["Revenue"] == 1
    answer = assert_structured_answer(data, "missing")
    assert any(block["type"] == "table" for block in answer["blocks"])


def test_ask_group_by_aggregate_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "group-by.xlsx")

    response = ask(client, dataset_id, "What is total revenue by category?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "group_by_sum"
    assert data["operation"]["params"]["group_by"] == "Category"
    assert data["operation"]["params"]["value_column"] == "Revenue"
    assert result_rows_by_first_column(data["result"]) == {"C": 500.0, "A": 450.0, "B": 400.0}
    answer = assert_structured_answer(data)
    assert any(block["type"] == "table" for block in answer["blocks"])


def test_ask_average_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "average.xlsx")

    response = ask(client, dataset_id, "What is the average revenue?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "numeric_summary"
    revenue = result_row_map(data["result"], "Revenue")
    assert revenue["Mean"] == pytest.approx(sample_df["Revenue"].mean())
    answer = assert_structured_answer(data)
    assert any(block["type"] == "metrics" for block in answer["blocks"])


def test_ask_top_rows_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "top-rows.xlsx")

    response = ask(client, dataset_id, "Show me the top 3 rows by revenue.")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "top_n"
    assert data["operation"]["params"]["n"] == 3
    assert len(data["result"]["rows"]) == 3
    top_row = dict(zip(data["result"]["columns"], data["result"]["rows"][0]))
    assert top_row["Revenue"] == 500.0
    assert top_row["Customer"] == "Echo"
    answer = assert_structured_answer(data)
    table_blocks = [block for block in answer["blocks"] if block["type"] == "table"]
    assert table_blocks
    assert "Revenue" in table_blocks[0]["columns"]


def test_ask_correlation_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "correlation.xlsx")

    response = ask(client, dataset_id, "What is the correlation between revenue and orders?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "correlation"
    correlation_row = next(
        row for row in data["result"]["rows"] if set(row[:2]) == {"Revenue", "Orders"}
    )
    assert correlation_row[2] == pytest.approx(sample_df["Revenue"].corr(sample_df["Orders"]))
    answer = assert_structured_answer(data)
    assert any(block["type"] == "metrics" for block in answer["blocks"])


def test_ask_invalid_model_json_falls_back_to_safe_structured_answer(client, sample_df, monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            assert "dataset_context" in kwargs["input"]
            return type("FakeResponse", (), {"output_text": "<b>not json</b>"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    dataset_id, _ = upload_dataframe(client, sample_df, "invalid-model-json.xlsx")

    response = ask(client, dataset_id, "Summarize this dataset.")

    assert response.status_code == 200
    data = response.get_json()
    answer = assert_structured_answer(data, "column")
    assert data["operation"]["type"] == "column_summary"
    assert data["operation_results"][0]["status"] == "ok"
    assert "not json" not in json.dumps(answer).lower()


def test_ask_unsupported_question(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "unsupported.xlsx")

    response = ask(client, dataset_id, "Write me a poem.")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"] is None
    assert_structured_answer(data)
    assert "approved analysis operations" in data["answer_text"].lower()


def test_dangerous_prompt_is_read_only(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "readonly.xlsx")
    before = client.get(f"/api/datasets/{dataset_id}/preview").get_json()

    response = ask(client, dataset_id, "Delete all rows")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"] is None
    assert_structured_answer(data)
    assert "read-only" in data["answer_text"].lower() or "not modified" in data["answer_text"].lower()

    after = client.get(f"/api/datasets/{dataset_id}/preview").get_json()
    assert after == before
    assert len(after["preview"]) > 0


def test_dataset_isolation(client):
    first_df = pd.DataFrame(
        {
            "Customer": ["One", "Two"],
            "Category": ["Alpha", "Beta"],
            "Revenue": [10.0, 20.0],
            "Orders": [1, 2],
            "Date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        }
    )
    second_df = pd.DataFrame(
        {
            "Customer": ["Three", "Four"],
            "Category": ["Alpha", "Beta"],
            "Revenue": [1000.0, 2000.0],
            "Orders": [10, 20],
            "Date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
        }
    )
    first_id, _ = upload_dataframe(client, first_df, "first.xlsx")
    second_id, _ = upload_dataframe(client, second_df, "second.xlsx")

    first_response = ask(client, first_id, "What is total revenue by category?").get_json()
    second_response = ask(client, second_id, "What is total revenue by category?").get_json()

    first_result = result_rows_by_first_column(first_response["result"])
    second_result = result_rows_by_first_column(second_response["result"])
    assert first_result == {"Beta": 20.0, "Alpha": 10.0}
    assert second_result == {"Beta": 2000.0, "Alpha": 1000.0}
    assert first_result != second_result


def test_dataset_isolation_after_persistence(client):
    first_df = pd.DataFrame(
        {
            "Customer": ["One", "Two"],
            "Category": ["Alpha", "Beta"],
            "Revenue": [10.0, 20.0],
            "Orders": [1, 2],
            "Date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        }
    )
    second_df = pd.DataFrame(
        {
            "Customer": ["Three", "Four"],
            "Category": ["Alpha", "Beta"],
            "Revenue": [1000.0, 2000.0],
            "Orders": [10, 20],
            "Date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
        }
    )
    first_id, first_upload = upload_dataframe(client, first_df, "first-persisted.xlsx")
    second_id, second_upload = upload_dataframe(client, second_df, "second-persisted.xlsx")

    assert first_id != second_id
    assert first_upload["metadata"]["stored_filename"] != second_upload["metadata"]["stored_filename"]

    first_response = ask(client, first_id, "What is total revenue by category?")
    second_response = ask(client, second_id, "What is total revenue by category?")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert result_rows_by_first_column(first_response.get_json()["result"]) == {"Beta": 20.0, "Alpha": 10.0}
    assert result_rows_by_first_column(second_response.get_json()["result"]) == {"Beta": 2000.0, "Alpha": 1000.0}


def test_missing_uploaded_file_returns_clean_error(client, app, sample_df):
    dataset_id, data = upload_dataframe(client, sample_df, "missing-file.xlsx")
    stored_path = Path(app.config["UPLOAD_FOLDER"]) / data["metadata"]["stored_filename"]
    stored_path.unlink()

    response = client.get(f"/api/datasets/{dataset_id}/preview")

    assert response.status_code == 404
    response_data = response.get_json()
    assert "error" in response_data
    assert "missing" in response_data["error"].lower()


def test_column_names_with_spaces(client):
    df = pd.DataFrame(
        {
            "Customer Name": ["Acme", "Beta", "Core"],
            "Product Category": ["Hardware", "Software", "Hardware"],
            "Total Revenue": [100.0, 200.0, 300.0],
            "Order Count": [1, 2, 3],
        }
    )
    dataset_id, _ = upload_dataframe(client, df, "spaces.xlsx")

    response = ask(client, dataset_id, "What is total revenue by product category?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "group_by_sum"
    assert data["operation"]["params"]["group_by"] == "Product Category"
    assert data["operation"]["params"]["value_column"] == "Total Revenue"
    assert result_rows_by_first_column(data["result"]) == {"Hardware": 400.0, "Software": 200.0}


def test_large_xlsx_smoke(client):
    rows = 1000
    df = pd.DataFrame(
        {
            "Customer": [f"Customer {i}" for i in range(rows)],
            "Category": ["A" if i % 2 == 0 else "B" for i in range(rows)],
            "Revenue": [float(i + 1) for i in range(rows)],
            "Orders": [(i % 5) + 1 for i in range(rows)],
            "Date": pd.date_range("2026-01-01", periods=rows, freq="D"),
        }
    )
    dataset_id, _ = upload_dataframe(client, df, "large.xlsx")

    response = ask(client, dataset_id, "What is the total revenue?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "numeric_summary"
    revenue = result_row_map(data["result"], "Revenue")
    assert revenue["Sum"] == pytest.approx(df["Revenue"].sum())
