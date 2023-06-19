import csv
import tempfile

import openpyxl
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


# https://learn.microsoft.com/en-us/deployoffice/compat/office-file-format-reference
def xlsx_to_csv(file_path: str, sheet_name: str = None) -> list[str]:
    """
    Convert a workbook into a list of csv files
    :param file_path: the path to the workbook
    :param sheet_name: the name of the sheet to convert
    :return: a list of temporary file names
    """
    # Load the workbook and select the active worksheet
    wb = openpyxl.load_workbook(file_path)
    # ws = wb.active
    #
    # # Create a new temporary file and write the contents of the worksheet to it
    # with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False) as f:
    #     c = csv.writer(f)
    #     for r in ws.rows:
    #         c.writerow([cell.value for cell in r])
    # return f.name
    # load all sheets if sheet_name is None
    wb = wb if sheet_name is None else [wb[sheet_name]]
    temp_file_name = []
    # Iterate over the worksheets in the workbook
    for ws in wb:
        # Create a new temporary file and write the contents of the worksheet to it
        with tempfile.NamedTemporaryFile(mode="w+", newline="", suffix=".csv", delete=False) as f:
            c = csv.writer(f)
            for r in ws.rows:
                c.writerow([cell.value for cell in r])
            temp_file_name.append(f.name)
    # print(f'all Sheets are saved to temporary file {temp_file_name}')
    return temp_file_name


class XLSXLoader(BaseLoader):
    """Loads an XLSX file into a list of documents.

    Each document represents one row of the CSV file converted from the XLSX file.
    Every row is converted into a key/value pair and outputted to a new line in the
    document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: str | None = None,
        csv_args: dict | None = None,
        encoding: str | None = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> list[Document]:
        """Load data into document objects."""

        docs = []
        csv_files = xlsx_to_csv(self.file_path)
        for csv_file in csv_files:
            with open(csv_file, newline="", encoding=self.encoding) as csvfile:
                csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
                for i, row in enumerate(csv_reader):
                    content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                    try:
                        source = row[self.source_column] if self.source_column is not None else self.file_path
                    except KeyError:
                        raise ValueError(f"Source column '{self.source_column}' not found in CSV file.")
                    metadata = {"source": source, "row": i}
                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)

        return docs
