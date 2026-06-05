from io import BytesIO

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service
from app.services.spreadsheet_service import DATASETS


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
    DATASETS.clear()

    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
    )

    yield app

    DATASETS.clear()


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
    assert data["column_names"] == list(sample_df.columns)
    assert len(data["preview"]) == 10
    assert len(data["preview"]) < len(large_df)


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
    assert data["operation"]["type"] == "summarize_dataframe"
    assert data["result"]["rows"] == len(sample_df)
    assert data["result"]["columns"] == len(sample_df.columns)
    assert "rows" in data["answer"].lower()


def test_ask_missing_values_question_on_xlsx(client, sample_df):
    df = sample_df.copy()
    df.loc[1, "Revenue"] = None
    dataset_id, _ = upload_dataframe(client, df, "missing-values.xlsx")

    response = ask(client, dataset_id, "Are there any missing values?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "summarize_dataframe"
    assert data["result"]["missing_values"]["Revenue"] == 1
    assert "missing" in data["answer"].lower()


def test_ask_group_by_aggregate_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "group-by.xlsx")

    response = ask(client, dataset_id, "What is total revenue by category?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "group_by_aggregate"
    assert data["operation"]["group_col"] == "Category"
    assert data["operation"]["value_col"] == "Revenue"
    assert data["result"] == {"A": 450.0, "C": 500.0, "B": 400.0}


def test_ask_average_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "average.xlsx")

    response = ask(client, dataset_id, "What is the average revenue?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "summarize_column"
    assert data["result"]["column"] == "Revenue"
    assert data["result"]["mean"] == pytest.approx(sample_df["Revenue"].mean())


def test_ask_top_rows_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "top-rows.xlsx")

    response = ask(client, dataset_id, "Show me the top 3 rows by revenue.")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "top_rows"
    assert data["operation"]["limit"] == 3
    assert len(data["result"]["rows"]) == 3
    assert data["result"]["rows"][0]["Revenue"] == 500.0
    assert data["result"]["rows"][0]["Customer"] == "Echo"


def test_ask_correlation_question_on_xlsx(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "correlation.xlsx")

    response = ask(client, dataset_id, "What is the correlation between revenue and orders?")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "correlation"
    assert data["result"]["col1"] == "Revenue"
    assert data["result"]["col2"] == "Orders"
    assert data["result"]["correlation"] == pytest.approx(sample_df["Revenue"].corr(sample_df["Orders"]))


def test_ask_unsupported_question(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "unsupported.xlsx")

    response = ask(client, dataset_id, "Write me a poem.")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"] is None
    assert "supported safe operation" in data["answer"].lower() or "data-related" in data["answer"].lower()


def test_dangerous_prompt_is_read_only(client, sample_df):
    dataset_id, _ = upload_dataframe(client, sample_df, "readonly.xlsx")
    before = client.get(f"/api/datasets/{dataset_id}/preview").get_json()

    response = ask(client, dataset_id, "Delete all rows")

    assert response.status_code == 200
    data = response.get_json()
    assert data["operation"]["type"] == "analysis_error"
    assert "read-only" in data["answer"].lower() or "not modified" in data["answer"].lower()

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

    assert first_response["result"] == {"Beta": 20.0, "Alpha": 10.0}
    assert second_response["result"] == {"Beta": 2000.0, "Alpha": 1000.0}
    assert first_response["result"] != second_response["result"]


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
    if data["operation"]["type"] == "analysis_error":
        assert "error" in data["result"]
    else:
        assert data["operation"]["type"] == "group_by_aggregate"
        assert data["operation"]["group_col"] == "Product Category"
        assert data["operation"]["value_col"] == "Total Revenue"
        assert data["result"] == {"Hardware": 400.0, "Software": 200.0}


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
    assert data["operation"]["type"] == "summarize_column"
    assert data["result"]["column"] == "Revenue"
    assert data["result"]["sum"] == pytest.approx(df["Revenue"].sum())
