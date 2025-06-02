# MCLEAP

Computing the Delaunay triangulation of moving points in 2D in the GPU. 

## Getting Started

### Building

After cloning the repository, you have to download the submodules.

```
git clone git@github.com:hporro/mcleap.git
cd mcleap
git submodule update --init
mkdir build
cd build
cmake ..
```

### Running

Then you can run one of the test executables or experiment executables:

```
~mcleap/build$ ./tests/test_frnn
```
