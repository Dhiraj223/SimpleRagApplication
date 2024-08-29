from typing import List, Dict, Union

class DocumentSplitter:
    """
    A class for splitting text data into chunks.
    """

    def __init__(self, data: str, chunk_size: int = 1000, overlap: int = 200, add_start_index: bool = True):
        """
        Initialize the DocumentSplitter.

        Args:
            data (str): The text data to be split.
            chunk_size (int): The size of each chunk in characters.
            overlap (int): The number of overlapping characters between chunks.
            add_start_index (bool): Whether to include the start index in the output.
        """
        self.data = data
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.add_start_index = add_start_index

    def split_data(self) -> List[Union[str, Dict[str, Union[str, int]]]]:
        """
        Split the data into chunks.

        Returns:
            List[Union[str, Dict[str, Union[str, int]]]]: A list of chunks, either as strings or dictionaries with text and start index.
        """
        chunks = []
        data_length = len(self.data)
        start_index = 0

        while start_index < data_length:
            end_index = min(start_index + self.chunk_size, data_length)
            chunk = self.data[start_index:end_index]

            if self.add_start_index:
                chunks.append({
                    'text': chunk,
                    'start_index': start_index
                })
            else:
                chunks.append(chunk)

            start_index += self.chunk_size - self.overlap

        return chunks