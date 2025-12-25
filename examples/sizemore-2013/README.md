# HodgeRanking.py

A Python implementation of HodgeRank for ranking from pairwise comparisons, based on the work of Sizemore (2013) and others.

This folder is licensed under the [MIT License](LICENSE) (c) 2025 Yina Tang. See LICENSE for details.

## Dependencies

To run `HodgeRanking.py` and `sizemore-2013.py`, run the following command to install the required packages:

```sh
pip install numpy pandas matplotlib networkx
```

This code has been tested on VS Code 1.107.1 on macOS 14.7.1 with the following package versions:

```
Python 3.12.7
numpy 2.1.2
pandas 2.2.3
matplotlib 3.10.0
networkx 3.4.2
```

## Example Usage

`HodgeRanking.py` contains the main class `HodgeRanking`, which can be used to perform ranking and decomposition from pairwise comparison data.

`sizemore-2013.py` contains scripts that replicate the two small examples from Sizemore's thesis at pp. 82-86. 

To run example 1, use the command:

```sh
python sizemore-2013.py 1
```

The script saves the result ranking to a CSV file. It can also optionally draw the pairwise comparison matrix `Y` with the flag `--graph-y`: 

```sh
python sizemore-2013.py 1 --graph-y
```


## Reference

[1] Sizemore, R. K. (2013) Hodgerank: Applying combinatorial Hodge theory to sports ranking. MA Thesis. Wake Forest University, Winston-Salem, NC. Available at: https://wakespace.lib.wfu.edu/bitstream/handle/10339/38577/Sizemore_wfu_0248M_10444.pdf
