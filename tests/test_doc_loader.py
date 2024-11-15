# tests/test_doc_loader.py
import unittest
import os
from pathlib import Path
from zrag.doc_loader import DocumentLoader, Document, _process_file  # Import _process_file

class TestDocumentLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("./test_data")
        cls.test_dir.mkdir(exist_ok=True)
        cls.pdf_path = cls.test_dir / "test.pdf"
        cls.md_path = cls.test_dir / "test.md"
        cls.txt_path = cls.test_dir / "test.txt"
        cls.py_path = cls.test_dir / "test.py"

        # Create dummy files
        with open(cls.pdf_path, "wb") as f:
            f.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n" # Valid PDF header
                   b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
                   b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
                   b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n"
                   b"/Contents 4 0 R\n>>\nendobj\n"
                   b"4 0 obj\n<<\n/Length 21\n>>\nstream\nBT\n/F1 12 Tf\n72 712 Td\n(Test PDF)Tj\nET\nendstream\nendobj\n"
                   b"xref\n0 5\n0000000000 65535 f\n0000000018 00000 n\n0000000077 00000 n\n"
                   b"0000000178 00000 n\n0000000457 00000 n\ntrailer\n<<\n/Root 1 0 R\n/Size 5\n>>\nstartxref\n565\n%%EOF\n")
        with open(cls.md_path, "w", encoding="utf-8") as f:
            f.write("# Test Markdown\nThis is a test.")
        with open(cls.txt_path, "w", encoding="utf-8") as f:
            f.write("This is a test text file.")
        with open(cls.py_path, "w", encoding="utf-8") as f:
            f.write("def test():\n    print('Hello')")

    @classmethod
    def tearDownClass(cls):
       for file in [cls.pdf_path, cls.md_path, cls.txt_path, cls.py_path]:
            if file.exists():
                os.remove(file)
       if cls.test_dir.exists():
            os.rmdir(cls.test_dir)

    def test_load_single_pdf(self):
        loader = DocumentLoader(str(self.test_dir))
        # Use _process_file directly for more control in testing:
        documents = _process_file(self.pdf_path, loader.encoding, None, None, None, code_mime_types=loader.CODE_MIME_TYPES)  # Pass code_mime_types
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertIn("page_label", documents[0].metadata)

    def test_load_single_markdown(self):
        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(filenames=["test.md"])
        self.assertEqual(len(documents), 1)
        self.assertIn("Test Markdown", documents[0].text)

    def test_load_single_text(self):
        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(filenames=["test.txt"])
        self.assertEqual(len(documents), 1)
        self.assertIn("test text file", documents[0].text)

    def test_load_single_code(self):
        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(filenames=["test.py"])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "def test():\n    print('Hello')") # Check without preprocessing

    def test_load_with_preprocess(self):
        def preprocess(text):
            return text.upper()
        loader = DocumentLoader(str(self.test_dir))

        # Use _process_file directly for more control in testing:
        documents = _process_file(self.txt_path, loader.encoding, None, None, None, preprocess_fn=preprocess, code_mime_types=loader.CODE_MIME_TYPES) # Pass preprocess_fn and code_mime_types
        self.assertEqual(len(documents), 1)  # Check if a document was loaded
        self.assertEqual(documents[0].text, "THIS IS A TEST TEXT FILE.")

    def test_load_recursive(self):
        # Create a subdirectory and a file within it
        subdir = self.test_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        subfile = subdir / "subtest.txt"
        with open(subfile, "w", encoding="utf-8") as f:
            f.write("This is a subdirectory test file.")

        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(recursive=True)

        # Remove the created subdirectory and file after testing
        os.remove(subfile)
        os.rmdir(subdir)
        self.assertGreaterEqual(len(documents), 4) # includes files from the recursive search

    def test_load_with_ext_filter(self):
        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(ext=["*.md", "*.txt"])
        self.assertEqual(len(documents), 2)

    def test_load_with_exc_filter(self):
        loader = DocumentLoader(str(self.test_dir))
        documents = loader.load(exc=["*.pdf"])
        self.assertEqual(len(documents), 3)




if __name__ == "__main__":
    unittest.main()