from numbers import Number
from pydicom.dataset import FileDataset, Dataset


class Instance(FileDataset):
    slice_index: int
    tags: tuple[Number]
    tag_index: tuple[int]

    def __init__(self, dataset: Dataset):
        super().__init__("",
                         dataset=dataset,
                         file_meta=dataset.file_meta,
                         preamble=b"\0" * 128
                         )
        self.slice_index = None
        self.tags = None
        self.tag_index = None

    def set_slice_index(self, slice_index):
        self.slice_index = slice_index

    def set_tags(self, tags):
        self.tags = tags

    def set_tag_index(self, idx):
        self.tag_index = idx
