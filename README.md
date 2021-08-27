## Overview 
DisasseML is a machine learning model that simulates a [disassembler](https://www.wikiwand.com/en/Disassembler) by converting machine code into human-readable assembly code. 

## Generating the training set 
To create a training set with the OBJDUMP disassembler, use the command 
```shell 
tools/gen-train-auto -o <model-name>
```
while the following command will generate a training set with the [NDISASM](https://linux.die.net/man/1/ndisasm) diassembler: 
```shell 
tools/gen-train-auto -o <model-name>
```

The issue with using the NDISASM diassembler is reflective of the short-comings of the disasseML project as a whole. Namely, x86 machine code is variable-length and needs to be decoded from the correct starting address to be valid. Since NDISASM includes metadata in its disassembling process for instructions, if the last couple of bytes of metadata decode as the beginning of a long instruction, NDISASM will decode them as such. Obviously this is a substantial risk for the rest of the machine code; the current position may be in the middle of another instruction. For further discussion of this issue and possible solutions, see [this Stack Overflow post](https://stackoverflow.com/questions/47420776/using-ndisasm-in-files-of-different-architectures).

At present, the ``tools/gen-train-auto`` script lacks a preprocessing step, this will be added in due course. 

## Hyperparameter Optimization 
To tune the model's hyperparameters without training, we rely on a traditional gridsearch approach.  First create the file ``data/config.json`` with default values (as described in the Configuration above). A potential improvement for this implementation would be using [Talos](https://github.com/autonomio/talos) or other standardized tools. In particular, the gridsearch implementation adds complexity that this library could reduce. 

The parameter grid can be created as follows. Additional functionality could come from splitting the grid across multiple files with another script, but we do not provide it here. Further improvements could be made from other hyperparameter optimization protocols, such as [random searches](https://docs.ray.io/en/latest/tune/index.html) or Bayesian optimization layers (as provided as part of scikit-learn).

## Entire grid in single file with ``tune``
The command 
```shell
./tune <model name> 
```
creates a single parameter grid and writes it to ``data/config.json``. This is particularly useful when there are few parameters to search or if all parameters are related and can be searched in one go. 

## Configuration

## Dependencies 

The following are required to generate the training set: 

* `GNU` coreutils (`bash`, etc.)

* `GNU` binutils (`objdump`, for AT&T syntax or non-Intel assembly languages)

* `NASM` (`ndisasm`, for Intel syntax)

* `Python 3.6`

## Training and Running the model

The following were used to run the model.

### Necessary dependencies

The following are vital and the program won't run without them. Different versions may work though. These packages can all be installed using the Python pacakge manager, `pip`.

* `Python 3.6`

* `tensorflow-gpu 1.10.0` &ndash; machine learning backend (NB: there is a patch to be applied (see next section) which only applies to `TensorFlow 1.10.0`, consequently other versions of TensorFlow **will not work**)

* `numpy 1.14.5` &ndash; for CPU-based mathematics

* `h5py 2.8.0` &ndash; for saving and loading Keras models


## System-Specific Concerns  
Since this repo's structure contains a ``/src/`` folder with the code, VSCode may throw unresolved import errors. This can be fixed as described in [this post](https://github.com/microsoft/python-language-server/blob/master/TROUBLESHOOTING.md#unresolved-import-warnings). 

Some of the bash scripts in ``/tools/`` may require a change of permissions. See [this post](https://unix.stackexchange.com/questions/203371/run-script-sh-vs-bash-script-sh-permission-denied) for how to do it. This is partially the fault of Git itself since it is a simple content tracker, as discussed [here](https://stackoverflow.com/questions/39666585/does-git-store-the-read-write-execute-permissions-for-files). 