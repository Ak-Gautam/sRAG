# test_doc_loader.py
import unittest
import os
import tempfile
import shutil
from zrag.doc_loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.create_test_files()
        self.loader = DocumentLoader(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        with open(os.path.join(self.test_dir, "test.txt"), "w") as f:
            f.write("This is a test text file.")

    def test_load_text_file(self):
        documents = self.loader.load(ext=[".txt"])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "This is a test text file.")

    def test_load_specific_filenames(self):
        documents = self.loader.load(filenames=["test.txt"])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["file_name"], "test.txt")

    def test_recursive_loading(self):
        subdir = os.path.join(self.test_dir, "subdir")
        os.mkdir(subdir)
        with open(os.path.join(subdir, "subfile.txt"), "w") as f:
            f.write("This is a file in a subdirectory.")

        documents = self.loader.load(recursive=True)
        self.assertEqual(len(documents), 2)