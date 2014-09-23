#!/usr/bin/python

import os
import sys
import pylab

from perf import run_benchmark

fignum = 0

def plot_to_file(report, filename):
    global fignum
    fignum += 1
    pylab.figure(fignum)

    run_to_label = {
        "stl" : "C++ STL",
        "compute" : "Boost.Compute"
    }

    for run in report.samples.keys():
        x = []
        y = []

        for sample in report.samples[run]:
            x.append(sample[0])
            y.append(sample[1])

        pylab.plot(x, y, marker='o', label=run_to_label[run])

    pylab.xlabel("Size")
    pylab.ylabel("Time (ms)")
    pylab.legend(loc='upper left')
    pylab.savefig(filename)

if __name__ == '__main__':
    sizes = [pow(2, x) for x in range(10, 26)]
    algorithms = [
        "accumulate",
        "count",
        "inner_product",
        "merge",
        "partial_sum",
        "partition",
        "reverse",
        "rotate",
        "saxpy",
        "set_difference",
        "sort",
        "unique",
    ]

    try:
        os.mkdir("perf_plots")
    except OSError:
        pass

    for algorithm in algorithms:
        print "running '%s'" % (algorithm)
        report = run_benchmark(algorithm, sizes, ["stl"])
        plot_to_file(report, "perf_plots/%s_time_plot.png" % algorithm)

