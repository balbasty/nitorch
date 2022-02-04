Getting Started
===============


**Clone and install `nitorch`**

.. code-block:: console

        pip install git+https://github.com/balbasty/nitorch

        # Or, alternatively
        git clone git@github.com:balbasty/nitorch.git
        pip install ./nitorch

However, this only installs the core dependencies (torch and numpy). 
If you wish to automatically install dependencies used by, /e.g./, 
readers and writers, plotters and/or dataset loaders, you can specify 
the extra tags `io`, `plot`, `data`. Alternatively, the tag `all` 
combines all of these dependencies.

.. code-block:: console

        pip install git+https://github.com/balbasty/nitorch#egg=nitorch[all]

        # Or, alternatively
        git clone git@github.com:balbasty/nitorch.git
        pip install -e "./nitorch[all]"

You may then start using NITorch in a Python program:

.. code-block:: python

        import nitorch as ni
        # my cool script

Or use high-level tools from the command line:

.. code-block:: console

        nitorch --help


**Demo code**

The demo folder contains various Jupyter notebooks that showcase some of NITorch's funtionality. Just follow the instructions in the notebook to get started, or follow the links below to run an installation-free copy instantly in Google Colab:

* `Affine Registration <https://colab.research.google.com/drive/13eSBtEvAp1wIJD0Rlvq5Q9kJWnuEc7WI?usp=sharing>`_
* `Spatial Tools <https://colab.research.google.com/drive/1-dfCosj9XoesFt7byIhp84p2JMUuHxby?usp=sharing>`_


**Compiling C++/CUDA extensions**

By default, a pure PyTorch implementation is used. However, we also 
provide a version of our image resampling tools written in C++/CUDA, 
which provide a x10 speedup over the default version. Because it requires
a specific compilation environment, this is for advanced users only.

Note that these extensions are built _against_ PyTorch and therefore pin
the installation to the specific PyTorch version used for compilation. 
To force pip to use a specific version of PyTorch, it is advised to 
install it beforehand and call `pip install` with the option 
`--no-build-isolation`. 

#. (If GPU support needed) Install `cuda <https://developer.nvidia.com/cuda-toolkit-archive>` 
    
  * You need the *driver* and the toolkit (*compiler*, *headers* and *libraries*).
  * Follow instructions from the nvidia website based on your OS and the cuda version supported by your device.
  * See also: section **Troubleshooting**.

#. Install NITorch with compilation enabled:

  .. code-block:: console

          git clone git@github.com:balbasty/nitorch.git
          cd nitorch
          NI_COMPILED_BACKEND="C" pip install .

          # Or, alternatively (the version used is an arbitrary example)
          pip install torch==1.9.0+cu111
          NI_COMPILED_BACKEND="C" pip install --no-build-isolation .


**Compiling your own wheel**

#. Build a wheel file

  .. code-block:: console

          git clone git@github.com:balbasty/nitorch.git
          cd nitorch
          ./setup.py bdist_wheel
          # or alternatively
          # NI_COMPILED_BACKEND="C" ./setup.py bdist_wheel

  This will create a wheel file in a `dist/` directory:

  .. code-block:: console

          .
          ├── dist
          │   ├── nitorch-[*].whl

#. Install wheel file using `pip`

  .. code-block:: console

          pip install nitorch-[*].whl

Note that when `NI_COMPILED_BACKEND="C"` is used, NITorch becomes specific 
to an **OS**, a **Python version** and (if CUDA is enabled) a **CUDA version**. 
Since we link against libtorch, it is also specific to a **PyTorch version**
You must therefore be careful about what packages are present in your 
environment.


**Troubleshooting**

CUDA

* Different versions of the CUDA toolkit support different *compute 
  capability* versions (see: https://en.wikipedia.org/wiki/CUDA#GPUs_supported). 
  You should install a version of the toolkit that is compatible with the   
  compute capability of your device.

* The toolkit installer allows both the *driver* and the *toolkit*
  (compiler, headers, libraries) to be installed. The driver needs admin 
  priviledges to be installed, but the toolkit does not. Here's a way to 
  install the toolkit without admin priviledges (copied from 
  `here <https://forums.developer.nvidia.com/t/72087/6>`_):

  .. code-block:: console

          ./cuda_<VERSION>_linux.run --silent --toolkit --toolkitpath=<INSTALLPATH> --defaultroot=<INSTALLPATH>

* If your CUDA toolkit is installed in a non-standard location (*i.e.*, 
  different from `/usr/local/cuda`), use the environement 
  variable `CUDA_HOME` to help the setup script locate it:

  .. code-block:: console

          CUDA_HOME=<PATH_TO_CUDA> ./setup.py install

  However, note that `nvcc` should call the correct nvidia compiler. 
  Therefore, setup your path accordingly:

  .. code-block:: console

          export PATH="$CUDA_HOME/bin:$PATH"

* The nvidia compiler (`nvcc`) calls a host compiler (`gcc`, `clang`, ...). 
  If you wish to use a non-standard host compiler (*e.g.*, you are using 
  `gcc-8` instead of the native `gcc`), things might be trickier. 
  A solution could be to alias nvcc so that it uses the `-ccbin` option 
  by default. In your `~/.bashrc`, add:

  .. code-block:: console

          alias nvcc='nvcc -ccbin <PATH_TO_GCC_BIN>'
