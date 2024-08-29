import os
import chardet
import docx
import PyPDF2
import requests
import pytesseract
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from typing import Union, Dict, Any
from urllib.parse import urlparse

class DocumentLoader:
    """A class for loading data from various file formats and web pages."""

    def __init__(self, path: str):
        """
        Initialize the DocumentLoader with a file path or URL.

        Args:
            path (str): The path to the file to be loaded or a URL.
        """
        self.path = path
        self.data = None

    def load_data(self) -> Union[str, Dict[str, Any]]:
        """
        Load data from the file or URL based on its type.

        Returns:
            Union[str, Dict[str, Any]]: The loaded data as a string or dictionary.

        Raises:
            ValueError: If the file format is unsupported or the URL is invalid.
            IOError: If there's an error reading the file or loading the web page.
        """
        try:
            if self._is_url(self.path):
                return self._load_web_page(self.path)
            
            file_extension = os.path.splitext(self.path)[1].lower()

            loaders = {
                '.pdf': self._load_pdf,
                '.docx': self._load_docx,
                '.html': self._load_html,
                '.htm': self._load_html,
                '.txt': self._load_txt,
                '.md': self._load_txt,
                '.csv': self._load_csv,
                '.xls': self._load_excel,
                '.xlsx': self._load_excel,
                '.jpg': self._load_image,
                '.jpeg': self._load_image,
                '.png': self._load_image
            }

            loader = loaders.get(file_extension)
            if loader:
                self.data = loader()
                return self.data
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except IOError as e:
            raise IOError(f"Error reading file or loading web page: {e}")

    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _load_web_page(self, url: str) -> str:
        """
        Load text from a web page.

        Args:
            url (str): The URL of the web page to load.

        Returns:
            str: The text content of the web page.

        Raises:
            ValueError: If the web page couldn't be loaded.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            text = soup.get_text()
            return text
        except requests.RequestException as e:
            raise ValueError(f"Failed to load web page: {e}")

    def _load_pdf(self) -> str:
        """Load text from a PDF file."""
        try:
            text = ""
            with open(self.path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise IOError(f"Error reading PDF file: {e}")

    def _load_docx(self) -> str:
        """Load text from a DOCX file."""
        try:
            doc = docx.Document(self.path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            raise IOError(f"Error reading DOCX file: {e}")

    def _load_html(self) -> str:
        """Load text from an HTML file."""
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'lxml')
                text = soup.get_text()
            return text
        except Exception as e:
            raise IOError(f"Error reading HTML file: {e}")

    def _load_txt(self) -> str:
        """Load text from a TXT or MD file."""
        try:
            with open(self.path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(self.path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except Exception as e:
            raise IOError(f"Error reading text file: {e}")

    def _load_csv(self) -> str:
        """Load text from a CSV file."""
        try:
            df = pd.read_csv(self.path)
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise IOError(f"Error reading CSV file: {e}")

    def _load_excel(self) -> str:
        """Load text from an Excel file."""
        try:
            df = pd.read_excel(self.path)
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise IOError(f"Error reading Excel file: {e}")

    def _load_image(self) -> str:
        """Load text from an image file using OCR."""
        try:
            image = Image.open(self.path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise IOError(f"Error reading image file: {e}")