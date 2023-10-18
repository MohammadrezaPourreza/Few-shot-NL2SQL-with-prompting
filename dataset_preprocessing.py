import os
import glob
import re
import pandas as pd
from langchain.utilities.sql_database import SQLDatabase
from azure_openai import get_completion_4, get_embedding


def table_description_parser(database_dir, table_name):
    """
    Returns a description string for the given table.

    Args:
    - database_dir (str): Path to the directory containing the CSV files.
    - table_name (str): Name of the table to be processed.

    Returns:
    - str: Description of the table.
    """
    file_path = os.path.join(database_dir, f"{table_name}.csv")
    if not os.path.exists(file_path):
        return f"No CSV found for table: {table_name}"

    db_description = f""
    table_df = pd.read_csv(file_path, encoding='latin-1')

    for _, row in table_df.iterrows():
        try:
            if pd.notna(row[2]):
                col_description = re.sub(r'\s+', ' ', str(row[2]))
                val_description = re.sub(r'\s+', ' ', str(row[4]))
                if pd.notna(row[4]):
                    db_description += f"Column {row[0]}: column description -> {col_description}, value description -> {val_description}\n"
                else:
                    db_description += f"Column {row[0]}: column description -> {col_description}\n"
        except Exception as e:
            print(e)
            db_description += "No column description"

    db_description += "\n"
    return db_description


def get_table_names(database_dir):
    """
    Returns a list of table description files present in the given directory of the bird dataset.

    Args:
    - database_dir (str): Path to the directory containing the CSV files.

    Returns:
    - List[str]: List of table names.
    """
    csv_files = glob.glob(f"{database_dir}/*.csv")
    return [os.path.basename(file_path).replace(".csv", "") for file_path in csv_files]


def get_table_create_statement_with_sample(db_uri, table_name):
    db = SQLDatabase.from_uri("sqlite:///"+db_uri)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw([table_name])


def get_table_description(db_name, table_name, create_statement, annotated_columns_description):
    prompt = f"""
You are a helpful database inspection assistant you are given details about the database name and a specific table information within the database.
Your task is to generate a short description of the table (2-3 sentences long).

You are provided with the following: database name, table name, create table statement, sample of 3 rows and annotated information of each column in the table.

Database name: {db_name}
Table Name: {table_name}
Create Table Statement + Sample:
```{create_statement}
```

Columns annotated information:
```
{annotated_columns_description}```
Table description: """
    return get_completion_4(prompt)


def generate_general_table_descriptions(db_path: str = "dev/dev_databases"):
    # Create an empty dataframe with desired columns
    df = pd.DataFrame(columns=['db_name', 'table_name', 'description', 'embedding'])

    databases = sorted(glob.glob(f"{db_path}/*"))
    for db in databases:
        db_name = os.path.basename(db)
        db_uri = f"{db_path}/{db_name}/{db_name}.sqlite"
        db_descriptions_path = f"{db}/database_description"
        tables = get_table_names(db_descriptions_path)
        print(db_descriptions_path)
        for table in tables:
            create_statement = get_table_create_statement_with_sample(db_uri, table)
            annotated_columns_description = table_description_parser(db_descriptions_path, table)
            table_description = get_table_description(db_name, table, create_statement, annotated_columns_description)
            embedding = get_embedding(table_description)

            # Append data to the dataframe using pandas.concat
            new_row = pd.DataFrame({
                'db_name': [db_name],
                'table_name': [table],
                'description': [table_description],
                'embedding': [embedding]
            })
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv('dev_tables_description.csv', index=False)


if __name__ == "__main__":
    generate_general_table_descriptions()