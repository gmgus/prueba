# PDF Information Extractor

This project extracts information like date, subject, and recipient from PDF files.

## How to Use

1.  **Set the PDF source:** Open the `process_document.py` file and modify the `source` variable to the path of your PDF file. For example:
    ```python
    source = "path/to/your/pdf_file.pdf"
    ```
2.  **Run the script:** Execute the `process_document.py` script.
3.  **View the output:** The script will print the extracted information (date, subject, recipient) to the console.

## Dependencies

The project uses the following libraries:

*   pdfplumber
*   PyPDF2
*   regex

### Installation

You can install the dependencies using pip:

```bash
pip install pdfplumber PyPDF2 regex
```
