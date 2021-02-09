"""Registered transform writers

Classes should be registered in their implementation file by importing
writer_classes and appending them:
>>> from nitorch.io.volumes.writers import writer_classes
>>> writer_classes.append(MyClassThatCanWrite)

This file is kept empty to avoid all registered writers to be erased by
autoreloading the module.
"""

from ..writers import writer_classes as all_writer_classes

writer_classes = []
all_writer_classes.append(writer_classes)
