"""Specialization for volumes."""
from functools import wraps
from .readers import reader_classes as streamline_reader_classes
from .writers import writer_classes as streamline_writer_classes
from .. import loadsave


@wraps(loadsave.map)
def map(*args, reader_classes=None, **kwargs):
    reader_classes = reader_classes or streamline_reader_classes
    return loadsave.map(*args, reader_classes=reader_classes, **kwargs)


@wraps(loadsave.load)
def load(*args, reader_classes=None, **kwargs):
    reader_classes = reader_classes or streamline_reader_classes
    return loadsave.load(*args, reader_classes=reader_classes, **kwargs)


@wraps(loadsave.loadf)
def loadf(*args, reader_classes=None, **kwargs):
    reader_classes = reader_classes or streamline_reader_classes
    return loadsave.loadf(*args, reader_classes=reader_classes, **kwargs)


@wraps(loadsave.save)
def save(*args, writer_classes=None, **kwargs):
    writer_classes = writer_classes or streamline_writer_classes
    return loadsave.save(*args, writer_classes=writer_classes, **kwargs)


@wraps(loadsave.savef)
def savef(*args, writer_classes=None, **kwargs):
    writer_classes = writer_classes or streamline_writer_classes
    return loadsave.savef(*args, writer_classes=writer_classes, **kwargs)
