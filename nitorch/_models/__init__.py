"""Vendored, dependency-light pretrained model definitions.

Internal to nitorch (not part of the public API): submodules here provide
self-contained architectures and weight-loading code for third-party
pretrained models used as building blocks elsewhere in nitorch (e.g. as a
feature extractor for registration), without pulling in that model's own
(often heavier) upstream package as a dependency.
"""
