"""`savef` / `savef_new` must dispatch to subclass overrides (issue #171)."""
import pytest
from nitorch.io.streamlines.mapping import MappedStreamlines


class _Recorder(MappedStreamlines):
    calls = []

    def __init__(self):
        pass

    def save(self, file_like=None, **meta):
        type(self).calls.append(('save', file_like, meta))
        return self

    @classmethod
    def save_new(cls, streamlines, file_like, like=None, **meta):
        cls.calls.append(('save_new', streamlines, file_like, like, meta))
        return 'new'


def setup_function(_):
    _Recorder.calls.clear()


def test_savef_dispatches_to_subclass_save():
    obj = _Recorder()
    assert obj.savef('out.trk', foo=1) is obj
    assert _Recorder.calls == [('save', 'out.trk', {'foo': 1})]


def test_savef_new_dispatches_to_subclass_save_new():
    assert _Recorder.savef_new([0], 'out.trk', None, foo=1) == 'new'
    assert _Recorder.calls == [('save_new', [0], 'out.trk', None, {'foo': 1})]


def test_base_still_not_implemented():
    obj = MappedStreamlines.__new__(MappedStreamlines)
    with pytest.raises(NotImplementedError):
        obj.savef('x')
    with pytest.raises(NotImplementedError):
        MappedStreamlines.savef_new([], 'x')


def test_trk_savef_uses_trk_save():
    trk = pytest.importorskip('nitorch.io.streamlines.trk')
    called = []
    obj = trk.TrkStreamlines.__new__(trk.TrkStreamlines)

    class _Struct:
        def save(self, f):
            called.append(f)

    obj._struct = _Struct()
    obj.savef('out.trk')
    assert called == ['out.trk']
