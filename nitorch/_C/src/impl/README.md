# Portable (CPU/GPU) implementations

This directory contains files that implement
core algorithms in a device-independent manner.

To do so, I make use of a small sets of macros defined in `common.h`, 
as well as a few `#ifdef __CUDACC__` in the code.

The first algorithms I implemented were the interpolation/splatting 
functions (`grid_pull`, `_grid_push`), implemented in `pushpull_common.cpp`.
I made a few dumb-ish design choices that carry over to the more recently 
implemented algorithms.

The main implementation is in a class (`MyAgloImpl`) that holds navigators
(data pointers and strides) and a few options (e.g. boundary conditions, 
interpolation orders). The algorithm is implemented in const functions that 
deal with a single voxel. For performance, there are several specialized 
functions (for 1D, 2D, 3D and for common special cases such as 
"linear isotropic"). There is a loop function, different for CPU and GPU, 
that calls the "single voxel" function on a bunch of indices.

There is often another class (`MyAlgoAllocator`) that preprocesses some 
options, extracts the navigators from the Tensor objects, and finds out 
if we can navigate the data using only 32 bit offsets. This class is 
built first and allows the main algorithm to be dispatched to the most efficient
offset type (GPU code can be faster if navigation uses only 32 bit offsets).

The second batch of algorithms I implemented relate to the full multigrid solver
(`relax`, `regulariser`, `resize`). At first, I tried to be smart and used 
member function pointers instead of if/else cases to dispatch to the correct 
specialized implementation, but I painfully found out that cuda 
deals poorly with member function pointers. Now two dispatch systems coexist:
function pointers on CPU and a switch statement on GPU. The other difference is my `Impl` classes grew too
large and could not be passed to the kernel by copy anymore (cuda kernels
cannot take more than 256 bytes of arguments). So now I manually move 
the object to device and give a pointer to the object to the kernel.

The main change compared to SPM is that the "field" (i.e. non-"grid") version
accepts channel-specific parameters for absolute/membrane/bending whereas JA's
version only takes scalars, plus a common channel-specific modulation. 
The application I have in mind involves jointly estimating images and smooth 
fields, where the images would use a membrane energy and the smooth fields a
bending energy. The other difference is that my code accepts channel-specific
maps to modulate the energies on a voxel-wise manner. I use them to implement 
TV-like penalties by iterative reweighting.

Lately, I have started templating the reduction type (which used to be 
double hardcoded). It's still fixed to double at instantiation, but on the 
GPU we could easily try to use float when the data is float. If it's stable
enough, it might give some performance gains.

**WARNING:** I had troubles because of a completely unexpected segfault
which took me *a day* to understand. What happened is I store a couple of 
arrays to store channel-related stuff in my objects. These arrays have a 
statically fixed size so that they use stack space. This size was arbitrarily 
fixed to 1024 (I think copied form the `sample_grid` code in pytorch,
whereas JA sets 128 in SPM). This causes my objects to be way too big 
and I think it causes weird side effects because (I think...) they take 
up all the stack. But the stack does not say "out of memory". Amyway,
Iwent back to max 128 channels and the issue seems resolved. 

**NOTE:** I had issues with some of my functions using massive amounts 
of VRAM that were never freed. It was weird because I never explicitly 
allocated data (I always relied on std or torch reference counting).
It seems to be caused by my code using more stack than available local 
memory (local memory is the private memory that belongs to each thread),
which was compensated by using some of the global (heap) memory. But a 
CUDA bug caused this memory to be never freed (or maybe it reserved 
that memory for future kernel calls that would require large stacks).
I found a semi-official answer by nvidia on their website giving a 
hacky solution: https://forums.developer.nvidia.com/t/61314/2. <br />
It would be better to try to reduce the stack size used by each thread. 
I could precompute less stuff, which would mean more ops but less stack
(maybe we don't care that much on the GPU).
