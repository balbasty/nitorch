import torch
import nitorch
import nitorch.spatial
import nitorch._C.spatial

torch.set_num_threads(10)

# shape = (128, 128, 128)
dim = 3
shape = (3, ) * dim

device = 'cuda'
torch.cuda.set_device(0)
torch.cuda.init()
dtype = torch.double

def make_data(shape, device, dtype):
    F = torch.nn.functional
    dim = len(shape)
    if dim == 3:
        mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]],
                           device=device, dtype=dtype)
    else:
        mat = torch.tensor([[[1, 0, 0], [0, 1, 0]]],
                           device=device, dtype=dtype)
    fov2vox = nitorch.spatial.fov2vox(shape, False).to(device=device, dtype=dtype)
    g = F.affine_grid(mat, (1, 1) + shape).to(device=device, dtype=dtype)
    g2 = g.matmul(fov2vox[:dim, :dim].transpose(0, 1)) \
       + fov2vox[:dim, dim].reshape((1, 1, 1, 1, dim))
    d = torch.rand((1,) + shape + (dim,), device=device, dtype=dtype)
    g3 = g + d

    src = torch.rand((1, 1) + shape, device=device, dtype=dtype)
    target = torch.rand((1, 1) + shape, device=device, dtype=dtype)
    return src, g3, target

print('--- Grad ---')
src, g3, target = make_data(shape, device, dtype)
src.requires_grad = True
g3.requires_grad = True
for (boundname, bound) in nitorch.spatial.BoundType.__members__.items():
    if boundname == 'sliding':
        continue
    for (intername, inter) in nitorch.spatial.InterpolationType.__members__.items():
        # if intername in ('nearest, linear'):
        #     continue
        if torch.autograd.gradcheck(nitorch.spatial.grid_grad, (src, g3, inter, bound, True),
                                    rtol=1., raise_exception=False):
            print('Bound: {:10} / Interpolation: {:10} / Success :D'.format(boundname, intername))
        else:
            print('Bound: {:10} / Interpolation: {:10} / Failure :('.format(boundname, intername))
        torch.cuda.empty_cache()

print('--- Pull ---')
src, g3, target = make_data(shape, device, dtype)
src.requires_grad = True
g3.requires_grad = True
for (boundname, bound) in nitorch.spatial.BoundType.__members__.items():
    if boundname == 'sliding':
        continue
    for (intername, inter) in nitorch.spatial.InterpolationType.__members__.items():
        if torch.autograd.gradcheck(nitorch.spatial.grid_pull, (src, g3, inter, bound, True),
                                    rtol=1., raise_exception=False):
            print('Bound: {:10} / Interpolation: {:10} / Success :D'.format(boundname, intername))
        else:
            print('Bound: {:10} / Interpolation: {:10} / Failure :('.format(boundname, intername))

print('--- Push ---')
src, g3, target = make_data(shape, device, dtype)
src.requires_grad = True
g3.requires_grad = True
for (boundname, bound) in nitorch.spatial.BoundType.__members__.items():
    if boundname == 'sliding':
        continue
    for (intername, inter) in nitorch.spatial.InterpolationType.__members__.items():
        if torch.autograd.gradcheck(nitorch.spatial.grid_push, (src, g3, shape, inter, bound, True),
                                    rtol=1., raise_exception=False):
            print('Bound: {:10} / Interpolation: {:10} / Success :D'.format(boundname, intername))
        else:
            print('Bound: {:10} / Interpolation: {:10} / Failure :('.format(boundname, intername))

print('--- Count ---')
src, g3, target = make_data(shape, device, dtype)
src.requires_grad = True
g3.requires_grad = True
for (boundname, bound) in nitorch.spatial.BoundType.__members__.items():
    if boundname == 'sliding':
        continue
    for (intername, inter) in nitorch.spatial.InterpolationType.__members__.items():
        if torch.autograd.gradcheck(nitorch.spatial.grid_count, (g3, shape, inter, bound, True),
                                    rtol=1., raise_exception=False):
            print('Bound: {:10} / Interpolation: {:10} / Success :D'.format(boundname, intername))
        else:
            print('Bound: {:10} / Interpolation: {:10} / Failure :('.format(boundname, intername))



# src, g3, target = make_data(shape, device, dtype)
# src.requires_grad = True
# g3.requires_grad = True
# ii = nitorch.spatial.InterpolationType.linear
# bb = nitorch.spatial.BoundType.replicate
# out = nitorch._C.spatial.grid_push(src, g3, shape, [bb], [ii], True)
# torch.autograd.gradcheck(nitorch.spatial.grid_push, (src, g3, shape, 'cubic', 'replicate', True),
#                           rtol=1., raise_exception=True)

# grad = torch.cat((src[...,None],src[...,None]), 4)
# out = nitorch._C.spatial.grid_pull_backward(src, src, g3, [bb], [ii], True)