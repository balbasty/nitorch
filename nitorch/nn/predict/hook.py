import torch
from nitorch.nn.base import Module


class HookWrapper(Module):
    """
    Model wrapper to instead return given layer as output, for e.g. self-supervised learning or t-SNE viz.
    Adapted from lucidrains: https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
    """
    def __init__(self, model, layer, proj=None):
        super().__init__()
        self.layer = layer
        self.model = model
        self.proj = proj
        self.hidden = {self.layer:[]}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.model.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.model.children()]
            return children[self.model]
        return None

    def _hook(self, _, input, output):
        """
        Behaviour of hook with current hypernetworks is different - using for-loop for unique generated weights in batch.
        Hence each iter in loop creates a unique hook.
        Using torch.cat() to append each new item seems to keep the graph intact for experiments using SimCLR and U-Net encoder.
        """
        if self.layer in self.hidden.keys() and type(self.hidden[self.layer]) == torch.Tensor:
            self.hidden[self.layer] = torch.cat([self.hidden[self.layer], output])
        else:
            self.hidden[self.layer] = output
    
    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        self.handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def _remove_hook(self):
        # remove hook handle - important when using e.g. mix of contrastive learning and reconstruction task
        self.handle.remove()
        self.hook_registered = False
    
    def forward(self, x, **kwargs):
        if self.layer == -1:
            return self.model(x, **kwargs)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.model(x, **kwargs)
        hidden = self.hidden[self.layer]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        if self.proj:
            hidden = self.proj(hidden)
        return hidden
