# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the tools used by the database agent."""

import datetime
import logging
import os
import re

from data_science.utils.utils import get_env_var
from google.adk.tools import ToolContext
from google.cloud import bigquery
from google.genai import Client

try:  # Optional: only needed for the Postgres GRANT helper (ensure_table_read_access)
    import psycopg2
    from psycopg2 import sql as psycopg2_sql
except ImportError:  # pragma: no cover
    psycopg2 = None
    psycopg2_sql = None

from .chase_sql import chase_constants

# Assume that `BQ_PROJECT_ID` is set in the environment. See the
# `data_agent` README for more details.
project = os.getenv("BQ_PROJECT_ID", None)
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
llm_client = Client(vertexai=True, project=project, location=location)

MAX_NUM_ROWS = 80


database_settings = None
bq_client = None


def get_bq_client():
    """Get BigQuery client."""
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=get_env_var("BQ_PROJECT_ID"))
    return bq_client


def get_database_settings():
    """Get database settings."""
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def update_database_settings():
    """Update database settings."""
    global database_settings
    datasets = get_bq_client().list_datasets()
    dataset_ids = [dataset.dataset_id for dataset in datasets]
    
    ddl_schema = get_bigquery_schema(
        dataset_id = dataset_ids,
        client=get_bq_client(),
        project_id=get_env_var("BQ_PROJECT_ID"),
    )
    database_settings = {
        "bq_project_id": get_env_var("BQ_PROJECT_ID"),
        "bq_dataset_id": dataset_ids,
        "bq_ddl_schema": ddl_schema,
        # Include ChaseSQL-specific constants.
        **chase_constants.chase_sql_constants_dict,
    }
    return database_settings


def get_bigquery_schema(dataset_id, client=None, project_id=None):
    """Retrieves schema and generates DDL with example values for a BigQuery dataset.

    Args:
        dataset_id (list): The IDs of the BigQuery dataset (e.g., ['my_dataset, 'second_dataset']).
        client (bigquery.Client): A BigQuery client.
        project_id (str): The ID of your Google Cloud Project.

    Returns:
        str: A string containing the generated DDL statements.
    """

    if client is None:
        client = bigquery.Client(project=project_id)

    # dataset_ref = client.dataset(dataset_id)
    #dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    ddl_statements = ""
    for dataset_ids in dataset_id:
        if dataset_ids != 'pso_ssm_usage':
            dataset_ref = client.dataset(dataset_ids)
            tables = list(client.list_tables(dataset_ref))

            for table in tables:
                table_ref = dataset_ref.table(table.table_id)
                table_obj = client.get_table(table_ref)

                # Check if table is a view
                if table_obj.table_type != "TABLE":
                    continue

                ddl_statement = f"CREATE OR REPLACE TABLE `{table_ref}` (\n"

                for field in table_obj.schema:
                    ddl_statement += f"  `{field.name}` {field.field_type}"
                    if field.mode == "REPEATED":
                        ddl_statement += " ARRAY"
                    if field.description:
                        ddl_statement += f" COMMENT '{field.description}'"
                    ddl_statement += ",\n"

                ddl_statement = ddl_statement[:-2] + "\n);\n\n"

                # Add example values if available (limited to first row)
                rows = client.list_rows(table_ref, max_results=5).to_dataframe()
                if not rows.empty:
                    ddl_statement += f"-- Example values for table `{table_ref}`:\n"
                    for _, row in rows.iterrows():  # Iterate over DataFrame rows
                        ddl_statement += f"INSERT INTO `{table_ref}` VALUES\n"
                        example_row_str = "("
                        for value in row.values:  # Now row is a pandas Series and has values
                            if isinstance(value, str):
                                example_row_str += f"'{value}',"
                            elif value is None:
                                example_row_str += "NULL,"
                            else:
                                example_row_str += f"{value},"
                        example_row_str = (
                            example_row_str[:-1] + ");\n\n"
                        )  # remove trailing comma
                        ddl_statement += example_row_str

                ddl_statements += ddl_statement

    return ddl_statements


def initial_bq_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    Args:
        question (str): Natural language question.
        tool_context (ToolContext): The tool context to use for generating the SQL
          query.

    Returns:
        str: An SQL statement to answer this question.
    """

    prompt_template = """
You are a BigQuery SQL expert tasked with answering user's questions about BigQuery tables by generating SQL queries in the GoogleSql dialect.  Your task is to write a Bigquery SQL query that answers the following question while using the provided context.

**Guidelines:**

- **Table Referencing:** Always use the full table name with the database prefix in the SQL statement.  Tables should be referred to using a fully qualified name with enclosed in backticks (`) e.g. `project_name.dataset_name.table_name`.  Table names are case sensitive.
- **Joins:** Join as few tables as possible. When joining tables, ensure all join columns are the same data type. Analyze the database and the table schema provided to understand the relationships between columns and tables.
- **Aggregations:**  Use all non-aggregated columns from the `SELECT` statement in the `GROUP BY` clause.
- **SQL Syntax:** Return syntactically and semantically correct SQL for BigQuery with proper relation mapping (i.e., project_id, owner, table, and column relation). Use SQL `AS` statement to assign a new name temporarily to a table column or even a table wherever needed. Always enclose subqueries and union queries in parentheses.
- **Column Usage:** Use *ONLY* the column names (column_name) mentioned in the Table Schema. Do *NOT* use any other column names. Associate `column_name` mentioned in the Table Schema only to the `table_name` specified under Table Schema.
- **FILTERS:** You should write query effectively  to reduce and minimize the total rows to be returned. For example, you can use filters (like `WHERE`, `HAVING`, etc. (like 'COUNT', 'SUM', etc.) in the SQL query.
- **LIMIT ROWS:**  The maximum number of rows returned should be less than {MAX_NUM_ROWS}.

**Schema:**

The database structure is defined by the following table schemas (possibly with sample rows):

```
{SCHEMA}
```

**Natural language question:**

```
{QUESTION}
```

**Think Step-by-Step:** Carefully consider the schema, question, guidelines, and best practices outlined above to generate the correct BigQuery SQL.

   """

    ddl_schema = tool_context.state["database_settings"]["bq_ddl_schema"]

    prompt = prompt_template.format(
        MAX_NUM_ROWS=MAX_NUM_ROWS, SCHEMA=ddl_schema, QUESTION=question
    )

    response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL"),
        contents=prompt,
        config={"temperature": 0.1},
    )

    sql = response.text
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()

    print("\n sql:", sql)

    tool_context.state["sql_query"] = sql

    return sql


def run_bigquery_validation(
    sql_string: str,
    tool_context: ToolContext,
) -> str:
    """Validates BigQuery SQL syntax and functionality.

    This function validates the provided SQL string by attempting to execute it
    against BigQuery in dry-run mode. It performs the following checks:

    1. **SQL Cleanup:**  Preprocesses the SQL string using a `cleanup_sql`
    function
    2. **DML/DDL Restriction:**  Rejects any SQL queries containing DML or DDL
       statements (e.g., UPDATE, DELETE, INSERT, CREATE, ALTER) to ensure
       read-only operations.
    3. **Syntax and Execution:** Sends the cleaned SQL to BigQuery for validation.
       If the query is syntactically correct and executable, it retrieves the
       results.
    4. **Result Analysis:**  Checks if the query produced any results. If so, it
       formats the first few rows of the result set for inspection.

    Args:
        sql_string (str): The SQL query string to validate.
        tool_context (ToolContext): The tool context to use for validation.

    Returns:
        str: A message indicating the validation outcome. This includes:
             - "Valid SQL. Results: ..." if the query is valid and returns data.
             - "Valid SQL. Query executed successfully (no results)." if the query
                is valid but returns no data.
             - "Invalid SQL: ..." if the query is invalid, along with the error
                message from BigQuery.
    """

    def cleanup_sql(sql_string):
        """Processes the SQL string to get a printable, valid SQL string."""

        # 1. Remove backslashes escaping double quotes
        sql_string = sql_string.replace('\\"', '"')

        # 2. Remove backslashes before newlines (the key fix for this issue)
        sql_string = sql_string.replace("\\\n", "\n")  # Corrected regex

        # 3. Replace escaped single quotes
        sql_string = sql_string.replace("\\'", "'")

        # 4. Replace escaped newlines (those not preceded by a backslash)
        sql_string = sql_string.replace("\\n", "\n")

        # 5. Add limit clause if not present
        if "limit" not in sql_string.lower():
            sql_string = sql_string + " limit " + str(MAX_NUM_ROWS)

        return sql_string

    logging.info("Validating SQL: %s", sql_string)
    sql_string = cleanup_sql(sql_string)
    logging.info("Validating SQL (after cleanup): %s", sql_string)

    final_result = {"query_result": None, "error_message": None}

    # More restrictive check for BigQuery - disallow DML and DDL
    if re.search(
        r"(?i)(update|delete|drop|insert|create|alter|truncate|merge)", sql_string
    ):
        final_result["error_message"] = (
            "Invalid SQL: Contains disallowed DML/DDL operations."
        )
        return final_result

    try:
        query_job = get_bq_client().query(sql_string)
        results = query_job.result()  # Get the query results

        if results.schema:  # Check if query returned data
            rows = [
                {
                    key: (
                        value
                        if not isinstance(value, datetime.date)
                        else value.strftime("%Y-%m-%d")
                    )
                    for (key, value) in row.items()
                }
                for row in results
            ][
                :MAX_NUM_ROWS
            ]  # Convert BigQuery RowIterator to list of dicts
            # return f"Valid SQL. Results: {rows}"
            final_result["query_result"] = rows

            tool_context.state["query_result"] = rows

        else:
            final_result["error_message"] = (
                "Valid SQL. Query executed successfully (no results)."
            )

    except (
        Exception
    ) as e:  # Catch generic exceptions from BigQuery  # pylint: disable=broad-exception-caught
        final_result["error_message"] = f"Invalid SQL: {e}"

    print("\n run_bigquery_validation final_result: \n", final_result)

    return final_result

def create_rules(dataset_id, table_id, column_name, client=None, project_id=None):
    """Prompts Gemini to create new rules for a specific column in a BigQuery table."""
    
    if not client:
        if not project_id:
            raise ValueError("Must provide either a BigQuery client or a project_id.")
        client = bigquery.Client(project=project_id)

    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    try:
        # Get table schema
        table = client.get_table(table_ref)
        schema = table.schema

        # Find the column in the schema
        column_schema = next((field for field in schema if field.name == column_name), None)
        if not column_schema:
            raise ValueError(f"Column '{column_name}' not found in table '{table_ref}'")

        # Sample data from the column
        query = f"SELECT `{column_name}` FROM `{table_ref}` WHERE `{column_name}` IS NOT NULL LIMIT 20"
        query_job = client.query(query)
        sample_rows = [row[column_name] for row in query_job.result()]
        
        # Prompt Gemini with schema and sample data
        prompt = f"""
        Given the following schema and sample data from a BigQuery table column,
        suggest data validation rules or quality checks that should be applied.

        Column: {column_name}
        Type: {column_schema.field_type}
        Mode: {column_schema.mode}
        Sample values: {sample_rows}

        Return rules in a bullet point list. 
        """

        response = llm_client.models.generate_content(
            model=os.getenv("BASELINE_NL2SQL_MODEL", "gemini-2.0-flash-001"),
            contents=prompt,
            config={"temperature": 0.1},
        )
        rules = response.text.strip()

        print(f"Suggested rules for {column_name} in {table_ref}:\n{rules}")
        return rules

    except Exception as e:
        print(f"Error while generating rules: {e}")
        return None


def ensure_table_read_access(conn, table_name, agent_user):
    """
    Ensures the agent_user has SELECT access on the given table.
    If not, grants the permission.

    Args:
        conn: A psycopg2 connection object.
        table_name (str): The name of the table (can include schema, e.g., 'public.control_table').
        agent_user (str): The database username or role to grant access to.
    """
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for ensure_table_read_access. "
            "Install it with `pip install psycopg2-binary`."
        )
    try:
        with conn.cursor() as cur:
            # Check if SELECT privilege exists
            cur.execute("""
                SELECT has_table_privilege(%s, %s, 'SELECT');
            """, (agent_user, table_name))
            has_access = cur.fetchone()[0]

            if has_access:
                print(f"[OK] '{agent_user}' already has SELECT access on '{table_name}'.")
            else:
                # Grant SELECT access
                cur.execute(psycopg2_sql.SQL("GRANT SELECT ON {} TO {};").format(
                    psycopg2_sql.Identifier(*table_name.split('.')),
                    psycopg2_sql.Identifier(agent_user)
                ))
                conn.commit()
                print(f"[FIXED] Granted SELECT access on '{table_name}' to '{agent_user}'.")
    except Exception as e:
        print(f"[ERROR] Failed to check/grant access: {e}")
        conn.rollback()
