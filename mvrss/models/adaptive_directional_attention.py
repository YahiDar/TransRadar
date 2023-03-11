import torch
from torch import nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
import torchvision.ops
# helper functions


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


# # following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
# class Deterministic(nn.Module):
#     def __init__(self, net):
#         super().__init__()
#         self.net = net
#         self.cpu_state = None
#         self.cuda_in_fwd = None
#         self.gpu_devices = None
#         self.gpu_states = None

#     def record_rng(self, *args):
#         self.cpu_state = torch.get_rng_state()
#         if torch.cuda._initialized:
#             self.cuda_in_fwd = True
#             self.gpu_devices, self.gpu_states = get_device_states(*args)

#     def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
#         if record_rng:
#             self.record_rng(*args)

#         if not set_rng:
#             return self.net(*args, **kwargs)

#         rng_devices = []
#         if self.cuda_in_fwd:
#             rng_devices = self.gpu_devices

#         with torch.random.fork_rng(devices=rng_devices, enabled=True):
#             torch.set_rng_state(self.cpu_state)
#             if self.cuda_in_fwd:
#                 set_device_states(self.gpu_devices, self.gpu_states)
#             return self.net(*args, **kwargs)



# class ReversibleBlock(nn.Module):
#     def __init__(self, f, g):
#         super().__init__()
#         self.f = Deterministic(f)
#         self.g = Deterministic(g)

#     def forward(self, x, f_args = {}, g_args = {}):
#         x1, x2 = torch.chunk(x, 2, dim = 1)
#         y1, y2 = None, None

#         with torch.no_grad():
#             y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
#             y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

#         return torch.cat([y1, y2], dim = 1)

#     def backward_pass(self, y, dy, f_args = {}, g_args = {}):
#         y1, y2 = torch.chunk(y, 2, dim = 1)
#         del y

#         dy1, dy2 = torch.chunk(dy, 2, dim = 1)
#         del dy

#         with torch.enable_grad():
#             y1.requires_grad = True
#             gy1 = self.g(y1, set_rng=True, **g_args)
#             torch.autograd.backward(gy1, dy2)

#         with torch.no_grad():
#             x2 = y2 - gy1
#             del y2, gy1

#             dx1 = dy1 + y1.grad
#             del dy1
#             y1.grad = None

#         with torch.enable_grad():
#             x2.requires_grad = True
#             fx2 = self.f(x2, set_rng=True, **f_args)
#             torch.autograd.backward(fx2, dx1, retain_graph=True)

#         with torch.no_grad():
#             x1 = y1 - fx2
#             del y1, fx2

#             dx2 = dy2 + x2.grad
#             del dy2
#             x2.grad = None

#             x = torch.cat([x1, x2.detach()], dim = 1)
#             dx = torch.cat([dx1, dx2], dim = 1)

#         return x, dx

# class IrreversibleBlock(nn.Module):
#     def __init__(self, f, g):
#         super().__init__()
#         self.f = f
#         self.g = g

#     def forward(self, x, f_args, g_args):
#         x1, x2 = torch.chunk(x, 2, dim = 1)
#         y1 = x1 + self.f(x2, **f_args)
#         y2 = x2 + self.g(y1, **g_args)
#         return torch.cat([y1, y2], dim = 1)

# class _ReversibleFunction(Function):
#     @staticmethod
#     def forward(ctx, x, blocks, kwargs):
#         ctx.kwargs = kwargs
#         for block in blocks:
#             x = block(x, **kwargs)
#         ctx.y = x.detach()
#         ctx.blocks = blocks
#         return x

#     @staticmethod
#     def backward(ctx, dy):
#         y = ctx.y
#         kwargs = ctx.kwargs
#         for block in ctx.blocks[::-1]:
#             y, dy = block.backward_pass(y, dy, **kwargs)
#         return dy, None, None

# class ReversibleSequence(nn.Module):
#     def __init__(self, blocks, ):
#         super().__init__()
#         self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

#     def forward(self, x, arg_route = (True, True), **kwargs):
#         f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
#         block_kwargs = {'f_args': f_args, 'g_args': g_args}
#         x = torch.cat((x, x), dim = 1)
#         x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
#         return torch.stack(x.chunk(2, dim = 1)).mean(dim = 0)
    

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn, dim = 64, kernel_size = 3):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation
        self.deform = DeformableConv2d(dim, dim, (kernel_size,kernel_size), padding = (kernel_size//2,kernel_size//2))
            
    def forward(self, x, **kwargs):
        x = self.deform(x)
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)
        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False) #[B*W, H, C] > #[B,W, H, C] > [B, C , H ,W]
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out



class ADA(nn.Module):
    def __init__(self, dim, depth, 
                 heads = 8, dim_heads = None, 
                 dim_index = 1, deform_k = [3, 3, 3, 3, 3, 3, 3, 3]):
        super().__init__()
        
        permutations = calculate_permutations(2, dim_index)

        
        self.pos_emb =  nn.Identity()

        layers = nn.ModuleList([])
        for mb in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation,  PreNorm(dim, SelfAttention(dim, heads, dim_heads)), dim = dim, kernel_size = deform_k[mb]) for permutation in permutations])
            layers.append(attn_functions)  

        self.layers = Sequential(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)