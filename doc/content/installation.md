# Installing Marlin

## Linux and INL HPC

Use the `moose-dev-openmpi-cudatorch-x86_64` [apptainer](https://mooseframework.inl.gov/help/inl/apptainer.html).

## macOS Arm64

**Conda:** create a separate conda environment for PyTorch (and do not activate it)

```
conda create -n pytorch_marlin pytorch
```

Check out and update the MOOSE submodule

```
git submodule update --init moose
```

Use the MOOSE configure script to enable libtorch and point it to the pytorch conda instalaltion.

```
cd moose
export LIBTORCH_DIR=$(conda activate pytorch_marlin; echo $CONDA_PREFIX; conda deactivate)
./configure --with-libtorch
```

Step back into the marlin root directory and build.

```
cd ..
make -j
```

Marlin on arm64 macs installed this way will support `cpu` and `mps` as compute devices. An mps/cpu (12 threads) runtime comparison for an M2 max laptop is shown below. Note that MPS only supports single precision (32bit) floating point numbers.

| Example | `cpu` runtime | `mps` runtime |
| - | - | - |
| `cahnhilliard.i` | 128s | 60s|
| `cahnhilliard2.i` | ~26000s | 1068s |
| `cahnhilliard3.i` | 440s | 29s |
| `rotating_grain.i`| 54s | 21s |

## Aurora (Intel OneAPI HPC)

Clone the repository and set up the build environment

```
git clone https://github.com/idaholab/marlin
cd marlin
MARLIN_DIR=$(pwd)
git submodule update --init moose

module load cmake
module load frameworks
module load petsc

python -m venv venv --system-site-packages
source ./venv/bin/activate
pip install packaging pyaml jinja2

export MOOSE_JOBS=6 METHODS=opt

export LIBTORCH_DIR=$(python -c 'import torch; print(torch.__path__[0])')
export INTEL_TORCH=${LIBTORCH_DIR/torch/intel_extension_for_pytorch}
```

Next patch libmesh `configure` because HDF5 detection is faulty on Aurora

```
cd moose
git submodule update --init libmesh
cd libmesh
git apply <<'EOF'
diff --git a/configure b/configure
index e28176191b..74b3c30977 100755
--- a/configure
+++ b/configure
@@ -61366,13 +61366,7 @@ return H5Fopen ();
   return 0;
 }
 _ACEOF
-if ac_fn_c_try_link "$LINENO"
-then :
-  ac_cv_lib_hdf5_H5Fopen=yes
-else case e in #(
-  e) ac_cv_lib_hdf5_H5Fopen=no ;;
-esac
-fi
+ac_cv_lib_hdf5_H5Fopen=yes
 rm -f core conftest.err conftest.$ac_objext conftest.beam \
     conftest$ac_exeext conftest.$ac_ext
 LIBS=$ac_check_lib_save_LIBS ;;
EOF
```

Continue building the prerequisites

```
cd ../scripts/
./update_and_rebuild_libmesh.sh --disable-xdr-required --disable-xdr --enable-hdf5-required --disable-petsc-hypre-required
./update_and_rebuild_wasp.sh
./update_and_rebuild_neml2.sh

cd ..
./configure --with-libtorch --with-neml2=$(pwd)/framework/contrib/neml2/installed/moose
```

Now build Marlin with custom `LDFLAGS` to ensure that the torch libraries are found (we use `-Wl,--disable-new-dtags` to set the `RPATH` rather than the `RUNPATH` to ensure +transitive+ dependencies are resolved)

```
cd ..
make CXXFLAGS="-fiopenmp" LDFLAGS="-fiopenmp -lintel-ext-pt-gpu -Wl,--disable-new-dtags -Wl,-rpath,${INTEL_TORCH}/lib -L${INTEL_TORCH}/lib -Wl,-rpath,${LIBTORCH_DIR}/lib" -j32
```

To run the code make sure each time your environment in `MARLIN_DIR` is set up with

```
module load cmake
module load frameworks
module load petsc
source ./venv/bin/activate
```

### Grab an interactive node for testing

```
qsub -I -A your_project -l walltime=0:10:00 -l filesystems=home
```
