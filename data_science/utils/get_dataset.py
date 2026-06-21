from google.cloud import bigquery

client = bigquery.Client(project="your-gcp-project-id")

try:
    dataset = client.get_dataset("your-gcp-project-id.sprint")
    print(" Sprint dataset is accessible.")
except Exception as e:
    print(" Error accessing sprint dataset:", e)
