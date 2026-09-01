from __future__ import annotations

import io

import pandas as pd

import scrape_sina_csi300_components_ascii as scraper


def select_component_table(html: str, expected_columns: list[str]) -> pd.DataFrame:
    expected = set(expected_columns)
    for table in pd.read_html(io.StringIO(html), displayed_only=False, header=None):
        columns = {str(column).strip() for column in table.columns}
        if expected.issubset(columns):
            return table.copy()
        if table.empty:
            continue
        first_row = [str(value).strip() for value in table.iloc[0].tolist()]
        if not expected.issubset(set(first_row)):
            continue
        positions = {column: first_row.index(column) for column in expected_columns}
        return pd.DataFrame(
            {
                column: table.iloc[1:, position].reset_index(drop=True)
                for column, position in positions.items()
            }
        )
    raise ValueError(f"Component table not found; expected={expected_columns}")


scraper.select_component_table = select_component_table


if __name__ == "__main__":
    scraper.main()
