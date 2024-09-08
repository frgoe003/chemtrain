# Connector

The connector compiles the JAX model to HLO using python and provides an interface to
evaluate the model in C++ via a shared library.


## Compilation

To compile the binary for testing:

```bash
bazel build -c opt --experimental_repo_remote_exec :main
```

