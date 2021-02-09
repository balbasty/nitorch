"""Registered transform readers

Classes should be registered in their implementation file by importing
reader_classes and appending them:
>>> from nitorch.io.volumes.readers import reader_classes
>>> reader_classes.append(MyClassThatCanRead)

This file is kept empty to avoid all registered readers to be erased by
autoreloading the module.
"""

from ..readers import reader_classes as all_reader_classes

reader_classes = []
all_reader_classes.append(reader_classes)
