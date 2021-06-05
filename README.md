# Community
![image](https://user-images.githubusercontent.com/16082928/120901139-5798cf80-c639-11eb-8ccd-460fa25a577c.png)


This program tests the perforamnce difference of multiple community detection algorithms on different kinds of generated graphs. It uses Lancichinetti–Fortunato–Radicchi benchmark graphs (see: "Benchmark graphs for testing community detection algorithms" by Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi, https://arxiv.org/pdf/0805.4770.pdf) and a modification using the Barabási–Albert preferential attachmentmodel (Barabasi and Albert 2002). This model still uses the LFR rewriting steps to maintain a powerlaw distributed community size.

# Usage
The code automaticly tests multiple configurations of graphs and reports the performance of community detection algorithms by comparing the found communities with the actual communities that have been generated in the benchmark graphs. The normalized mutual information score is used as a measure of similarity.

You can change which tests are run by changing the function calls at the end of main.py.

## Build

Use the included build.sh script to build the c++ code and to also install the required python packages. Run
`sudo chmod +x build.sh` and `./build.sh` to do the compilation and installation.


Now you can run the code by running `python3 main.py`.
