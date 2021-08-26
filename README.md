
### Hyperparameter Optimization 
To tune the model's hyperparameters without training, we rely on a traditional gridsearch approach.  First create the file ``data/config.json`` with default values (as described in the Configuration above). A potential improvement for this implementation would be using (Talos)[https://github.com/autonomio/talos] or other standardized tools. In particular, the gridsearch implementation adds complexity that this library could reduce. 

The parameter grid can be created in two ways. 

#### Entire grid in single file with ``tune``
The command 
```shell
./tune <model name> 
```
creates a single parameter grid and writes it to ``data/config.json``. This is particularly useful when there are few parameters to search or if all parameters are related and can be searched in one go. 

#### Separate grids with ``autotune``


If there are no grids for the gridsearch, the must be created. We provide some scripts to automate the process. 

### General Comments 
Since this repo's structure contains a ``/src/`` folder with the code, VSCode may throw unresolved import errors. This can be fixed as described in [this post](https://github.com/microsoft/python-language-server/blob/master/TROUBLESHOOTING.md#unresolved-import-warnings). 