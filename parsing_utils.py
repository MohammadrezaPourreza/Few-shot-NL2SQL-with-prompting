import json
from typing import List
from sqlglot import exp, parse_one
from sqlglot.dialects import Dialects


def get_query_tables(query: str) -> List[str]:
    # Returns all the table names which are used in the query
    parsed = parse_one(query, read=Dialects.SQLITE)
    return sorted(list({str(table.this).lower() for table in parsed.find_all(exp.Table)}))


def get_query_columns(query: str) -> List[str]:
    # Returns all the table names which are used in the query
    parsed = parse_one(query, read=Dialects.SQLITE)
    return sorted(list({str(table.this).lower() for table in parsed.find_all(exp.Column)}))


def process_tables_and_columns_in_dataset(dataset_path: str = "dev/dev.json"):
    # Read the JSON data from the file
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    # Iterate through all entries, get the tables and columns, and update the entry
    for entry in data:
        query = entry["SQL"]
        entry["tables"] = get_query_tables(query)
        entry["columns"] = get_query_columns(query)

    # Write the updated data back to the JSON file
    with open(dataset_path, 'w') as file:
        json.dump(data, file, indent=4)

