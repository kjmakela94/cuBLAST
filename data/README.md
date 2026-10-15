# Real matrices

Place `.mtx` or `.mtx.gz` files from SuiteSparse downloads here, then update
`MTX_PATHS` in the notebook. A `.mtx.gz` file can be read directly, but an outer
`.tar.gz` archive must be extracted first.

This repository does not download datasets automatically. Record each matrix's
name, SuiteSparse page, application domain, and license in your report. The three
synthetic matrices only demonstrate the workflow; they do not replace the three
real matrices required by the assignment.

Loading merges duplicate entries, removes explicit zeros, and sorts column
indices. SciPy expands symmetric Matrix Market storage. This baseline accepts
real-valued matrices with 32-bit CSR indices.

External files must be parsed before their working set can be checked. Check the
dimensions and memory requirements of very large files in advance. The configured
working-set budget is not a hard process memory limit.
