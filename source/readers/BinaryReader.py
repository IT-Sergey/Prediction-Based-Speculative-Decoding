import os
from collections import Counter


class BinaryReader:
    def __init__(self, files):
        self._files = files

    def read(self, max_bytes=None):
        """
        Read bytes from files, optionally limiting to a maximum number of bytes.

        Args:
            max_bytes (int, optional): Maximum total number of bytes to read.
                                      If None, reads all files completely.

        Returns:
            tuple: (Counter with symbol frequencies, bytes object with content)
        """
        total_symbol_counts = Counter()
        global_content = []
        total_bytes_read = 0

        for file_path in self._files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found at {file_path}. Skipping.")
                continue

            try:
                with open(file_path, 'rb') as f:
                    # If max_bytes is specified, calculate how many bytes to read from this file
                    if max_bytes is not None:
                        bytes_to_read = max_bytes - total_bytes_read
                        if bytes_to_read <= 0:
                            break  # Already read enough bytes

                        content = f.read(bytes_to_read)
                    else:
                        content = f.read()

                    total_symbol_counts.update(content)
                    global_content.append(content)

                    total_bytes_read += len(content)

                    if max_bytes is not None and total_bytes_read >= max_bytes:
                        break

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        return total_symbol_counts, b''.join(global_content)