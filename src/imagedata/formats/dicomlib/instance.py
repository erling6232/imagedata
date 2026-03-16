from numbers import Number
from pydicom.dataset import FileDataset, Dataset, FileMetaDataset
from typing import Any


class Instance(FileDataset):
    slice_index: int
    tags: tuple[Number]
    tag_index: tuple[int]

    def __init__(self, *args: Dataset, **kwargs: Any) -> None:
        if len(args) == 1:
            dataset = args[0]
        elif len(args) == 6:
            dataset = args[1]
        else:
            raise ValueError('Unexpected number of arguments ({}) for Instance.__init__'.format(len(args)))
        try:
            file_meta = dataset.file_meta
        except AttributeError:
            file_meta = FileMetaDataset()
        if 'file_meta' in kwargs:
            file_meta = kwargs['file_meta']
        preamble = b"\0" * 128
        if 'preamble' in kwargs:
            preamble = kwargs['preamble']
        super().__init__("",
                         dataset=dataset,
                         file_meta=file_meta,
                         preamble=preamble
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
