## Overview 
DisasseML is a machine learning model that simulates a [disassembler](https://www.wikiwand.com/en/Disassembler) by converting machine code into human-readable assembly code. Currently, the model has been trained to generate x86 assembly code using both AT&T and Intel syntaxes, although it could be extended further to other assembly languages, such as ARM. The project was inspired by [Andrew Davis' Black Hat 2015 talk](https://www.youtube.com/watch?v=LQh8dktQReI), although he focused primarily on malware identification, a potential application of the model implemented here. 

Recall that recurrent neural networks (RNNs) suffers from a lack of ability to handle long-term dependencies (vanishing gradient problem). In particular, when the gap between the relevant information and place where it's needed is small, RNNs may apply fairly well. Unfortunately, as this gap increases in size, RNNs are unable to connect relevant bits of information. [Bengio et al (1994)](https://ieeexplore.ieee.org/document/279181) explored the reasons behind RNNs' practical failures. This was also the subject of Hochreiter's 1991 thesis.

To avoid the long-term dependency issues of RNNs, this model instead depends on short long-term memory networks (LSTMs), which are capable of learning long-term dependencies. An intuitive explanation of the success of LSTMs can be read [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Of course [some](https://www.youtube.com/watch?v=S27pHKBEp30) might argue that the choice of LSTMs in this domain is second to transformers. 

## Generating the training set 
To create a training set with the OBJDUMP disassembler, use the command 
```shell 
tools/gen-train-auto -o <model-name>
```
while the following command will generate a training set with the [NDISASM](https://linux.die.net/man/1/ndisasm) diassembler: 
```shell 
tools/gen-train-auto -n <model-name>
```

The issue with using the NDISASM diassembler is reflective of the short-comings of the disasseML project as a whole. Namely, x86 machine code is variable-length and needs to be decoded from the correct starting address to be valid. Since NDISASM includes metadata in its disassembling process for instructions, if the last couple of bytes of metadata decode as the beginning of a long instruction, NDISASM will decode them as such. Obviously this is a substantial risk for the rest of the machine code; the current position may be in the middle of another instruction. For further discussion of this issue and possible solutions, see [this Stack Overflow post](https://stackoverflow.com/questions/47420776/using-ndisasm-in-files-of-different-architectures).

At present, the ``tools/gen-train-auto`` script lacks a preprocessing step, this will be added in due course. 

## Hyperparameter Optimization 
To tune the model's hyperparameters without training, we rely on a traditional gridsearch approach.  First create the file ``data/config.json`` with default values (as described in the Configuration section below). A potential improvement for this implementation would be using [Talos](https://github.com/autonomio/talos) or other standardized tools. In particular, the gridsearch implementation adds complexity that this library could reduce. 

The parameter grid can be created as follows. Additional functionality could come from splitting the grid across multiple files with another script, but we do not provide it here. Further improvements could be made from other hyperparameter optimization protocols, such as [random searches](https://docs.ray.io/en/latest/tune/index.html) or Bayesian optimization layers (as provided as part of [scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html)).

### Entire grid in single file with ``tune``
The command 
```shell
./tune <model name> 
```
creates a single parameter grid and writes it to ``data/config.json``. This is particularly useful when there are few parameters to search or if all parameters are related and can be searched in one go. 

One drawback of the current implementation is tha tif there are many, unrelated parameters to search, we do not provide a way to generate separate grids for the gridsearch. Finally, issues with regards speed and complexity in hyperparamter optimization could be solved by [hyperopt](https://hyperopt.github.io/hyperopt/). 

## Training, validating, and disassembling 
After selecting the hyperparameters, the model can be trained with the following command: 
```shell
./train <model name>
```
and can be validated on unseen data using 
```shell
python src/validator.py <model name>
```

In spite of the instruction boundary issues discussed earlier, the "alpha" version of the disassembler can be invoked using 
```shell 
./disasseml <model name> <binary> 
```
Note that valid model names here are ``att`` and ``intel``. 

To get a binary file from a program, OBJCOPY can be used as usual: 
```shell 
objcopy -O binary <input file> <output file>
```

## Configuration

DisasseML is configured using a JSON file stored in ``data/config.json``. For example, 

``data/config.json``

```javascript 
{
    // Maximum number of records during training.
    "max_records":               100000,
    // Maximum number of records during hyperparameter selection (gridsearch).
    "gs_records":                10000,
    // Overrideable model parameters.
    "model":{
        // Classifier performance metrics.
        "metrics":               ["accuracy"],
        // Input sequence length.
        "x_seq_len":             15,
        // Output sequence length.
        "y_seq_len":             64,
        // Mask value.
        "mask_value":            null,
        // Batch size.
        "batch_size":            100,
        // Number of training epochs.
        "epochs":                100,
        // Number of cross-validation folds.
        "kfolds":                10,
        // Whether to shuffle indices during cross-validation.
        "shuffle":               true,
        // Dimensionality of input space.
        "input_size":            256,
        // Number of hidden units per recurrent layer.
        "hidden_size":           256,
        // Dimensionality of the output space.
        "output_size":           128,
        // Type of recurrent unit (lstm, gru or rnn).
        "recurrent_unit":        "lstm",
        // Number of recurrent layers in the encoder.
        "encoder_layers":        1,
        // Number of recurrent layers in the decoder.
        "decoder_layers":        1,
        // Activation function to use in recurrent layers.
        "recurrent_activation":  "tanh",
        // Whether to use bias vectors in recurrent units.
        "recurrent_use_bias":    true,
        // Whether to use bias in the LSTM forget gate (has no effect if recurrent_unit is not "lstm").
        "recurrent_forget_bias": true,
        // Dropout rate between recurrent layers.
        "dropout":               0,
        // Dropout rate within recurrent sequences.
        "recurrent_dropout":     0,
        // Activation of the dense layer.
        "dense_activation":     "softmax",
        // Loss function.
        "loss":                 "categorical_crossentropy",
        // Optimizer.
        "optimizer":            "Adam",
        // Parameters to the optimizer.
        "opt_params": {
            // Learning rate.
            "lr": 0.001
        }
    },
    // Gridsearch parameter grid (inline). Listed values are combined and each combination overrides values in 'model'
    // during hyperparameter selection..
    "grid": {
        "hidden_size":    [64,128,256,512],
        "encoder_layers": [1,2,3],
        "decoder_layers": [1,2,3]
    }
}
```

## Dependencies 

The following are required to generate the training set: 

* `GNU` coreutils (`bash`, etc.)

* `GNU` binutils (`objdump`, for AT&T syntax or non-Intel assembly languages)

* `NASM` (`ndisasm`, for Intel syntax)

The following are vital and the model won't run without them. Different versions may work though. These packages can all be installed using the Python package manager.

* `Python 3.6`

* `tensorflow-gpu 1.10.0` &ndash; machine learning backend 

* `numpy 1.14.5` &ndash; for CPU-based mathematics

* `h5py 2.8.0` &ndash; for saving and loading Keras models

## System-Specific Concerns  
Since this repo's structure contains a ``/src/`` folder with the code, VSCode may throw unresolved import errors. This can be fixed as described in [this post](https://github.com/microsoft/python-language-server/blob/master/TROUBLESHOOTING.md#unresolved-import-warnings). 

Some of the bash scripts in ``/tools/`` may require a change of permissions. See [this post](https://unix.stackexchange.com/questions/203371/run-script-sh-vs-bash-script-sh-permission-denied) for how to do it. This is partially the fault of Git itself since it is a simple content tracker, as discussed [here](https://stackoverflow.com/questions/39666585/does-git-store-the-read-write-execute-permissions-for-files). 