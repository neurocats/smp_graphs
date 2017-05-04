

# smp\_graphs

Specifiy sensorimotor learning experiments as a graph of nodes and a
set of signals, corresponding nicely with the tapping approach.


## Items

-   x dict printing for dynamic reconf inspection

-   x base block

-   x two-pass init: complete by putting input init into second pass

-   loop block
    -   x parallel loop within graph, modify graph
    -   sequential loop for running block variations e.g hyperopt or evo

-   logging
    -   x std logging OK
    -   include git revision, initial and final config in log

-   x networkx for visualization?
    -   x standalone networkx graph from final config
    -   x graphviz

-   step, blocksize, ibuf
    -   min blocksize after pass 1
    -   how to optimize of min bs > 1?

-   sync / async block execution

-   minor stuff
    -   x print\_dict print fully compilable python code


## Examples

    cd smp_graph/experiments
    python experiment.py --conf conf/default2.conf
    python experiment.py --conf conf/default2_loop.conf


# Notes

This is my n-th attempt at designing a framework for computational
sensorimotor learning experiments. Earlier attempts include

-   **smp\_experiments**: defined config as name value pairs and some
    python code wrapping enabling the reuse of singular experiments
    defined elsewhere in an outer loop doing variations (collecting
    statistics, optimizing, &#x2026;)
-   **smpblocks**: first attempt at using plain python config files
    containing a dictionary specifying generic computation blocks and
    their connections. granularity was too small and specifying
    connections was too complicated
-   **smq**: tried to be more high-level, introducing three specific and
    fixed modules 'world', 'robot', 'brain'. Alas it turned out that
    left us too inflexible and obviosuly couldn't accomodate any
    experiments deviating from that schema. Is where we are ;)

