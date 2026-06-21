# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from google.cloud import bigquery
from pathlib import Path
from dotenv import load_dotenv

# Define and load .env file
env_file_path = Path(__file__).parent.parent.parent / ".env"
print(env_file_path)
load_dotenv(dotenv_path=env_file_path)


def load_csv_to_bigquery(project_id, dataset_name, table_name, csv_filepath):
    """Loads a CSV file into a BigQuery table."""
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_name)
    table_ref = dataset_ref.table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
    )

    with open(csv_filepath, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_name}.{table_name}")


def create_dataset_if_not_exists(project_id, dataset_name):
    """Creates a BigQuery dataset if it does not already exist."""
    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.{dataset_name}"

    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {dataset_id}")


def verify_dataset_accessibility(project_id, dataset_names):
    """Verifies that each dataset is accessible."""
    client = bigquery.Client(project=project_id)
    for dataset_name in dataset_names:
        dataset_id = f"{project_id}.{dataset_name}"
        try:
            client.get_dataset(dataset_id)
            print(f" Access verified for dataset: {dataset_id}")
        except Exception as e:
            raise RuntimeError(f" Failed to access dataset {dataset_id}: {e}")


def main():
    print(f"Current working directory: {os.getcwd()}")

    project_id = os.getenv("BQ_PROJECT_ID")
    if not project_id:
        raise ValueError("BQ_PROJECT_ID environment variable not set.")
    source_dataset = "source_data"
    quality_dataset = "sprint"

    source_csv_filepath = "data_science/utils/data/source.csv"
    control_csv_filepath = "data_science/utils/data/control.csv"

    # Verify access
    verify_dataset_accessibility(project_id, [source_dataset, quality_dataset])

    # Ensure datasets exist
    print("Ensuring datasets exist.")
    create_dataset_if_not_exists(project_id, source_dataset)
    create_dataset_if_not_exists(project_id, quality_dataset)

    # Load source data to source_data.source_table
    print("Loading source data.")
    load_csv_to_bigquery(project_id, source_dataset, "source_table", source_csv_filepath)

    # Load control data to sprint.control_table
    print("Loading data quality info.")
    load_csv_to_bigquery(project_id, quality_dataset, "control_table", control_csv_filepath)


if __name__ == "__main__":
    main()
