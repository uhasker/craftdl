# CraftDL

## Requirements

You will need `Python >= 3.9` and some recent version of `pip` (e.g. `pip >= 20.0`) 
to use this library.

## Installation

You can install CraftDL using the pip package manager:

```shell
pip install craftdl
```

## Matplotlib backend for plotting

The plotting facilities of CraftDL are built on top of `matplotlib`. Therefore you need
an appropriate `matplotlib` backend. If you encounter weird `matplotlib` errors when
trying to call a plotting function, you most probably don't have the appropriate backend
installed on selected.

Usually you want to use the `TkAgg` backend. You can use it like this:

```python
import matplotlib
matplotlib.use("TkAgg")
```
