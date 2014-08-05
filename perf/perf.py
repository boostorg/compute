#!/usr/bin/python

# driver script for boost.compute benchmarking. will run a
# benchmark for a given function (e.g. accumulate, sort).

import os
import sys
import random
import subprocess

try:
    import pylab
except:
    print 'pylab not found, no ploting...'
    pass

def run_perf_process(name, size, backend = ""):
    if not backend:
        proc = "perf_%s" % name
    else:
        proc = "perf_%s_%s" % (backend, name)

    try:
        output = subprocess.check_output(["./perf/" + proc, str(int(size))])
    except:
        return 0

    t = 0
    for line in output.split("\n"):
        if line.startswith("time:"):
            t = float(line.split(":")[1].split()[0])

    return t

class Report:
    def __init__(self, name):
        self.name = name
        self.samples = {}

    def add_sample(self, name, size, time):
        if not name in self.samples:
            self.samples[name] = []

        self.samples[name].append((size, time))

    def display(self):
        for name in self.samples.keys():
            print '=== %s with %s ===' % (self.name, name)
            print 'size,time (ms)'

            for sample in self.samples[name]:
                print '%d,%f' % sample

    def plot_time(self, name):
        if not name in self.samples:
            return

        x = []
        y = []

        any_valid_samples = False

        for sample in self.samples[name]:
            if sample[1] == 0:
                continue

            x.append(sample[0])
            y.append(sample[1])
            any_valid_samples = True

        if not any_valid_samples:
            return

        pylab.plot(x, y, marker='o', label=name)
        pylab.xlabel("Size")
        pylab.ylabel("Time (ms)")
        pylab.title(self.name)

    def plot_rate(self, name):
        if not name in self.samples:
            return

        x = []
        y = []

        any_valid_samples = False

        for sample in self.samples[name]:
            if sample[1] == 0:
                continue

            x.append(sample[0])
            y.append(float(sample[0]) / (float(sample[1]) * 1e-3))
            any_valid_samples = True

        if not any_valid_samples:
            return

        pylab.plot(x, y, marker='o', label=name)
        pylab.xlabel("Size")
        pylab.ylabel("Rate (values/s)")
        pylab.title(self.name)

def run_benchmark(name, sizes, vs=[]):
    report = Report(name)

    for size in sizes:
        time = run_perf_process(name, size)

        report.add_sample("compute", size, time)

    competitors = {
        "thrust" : ["accumulate",
                    "count",
                    "inner_product",
                    "partial_sum",
                    "sort",
                    "saxpy"],
        "tbb": ["accumulate",
                "merge",
                "sort"],
        "stl": ["accumulate",
                "count",
                "find_end",
                "includes",
                "inner_product",
                "is_permutation",
                "merge",
                "next_permutation",
                "nth_element"
                "partial_sum",
                "partition",
                "partition_point",
                "prev_permutation",
                "reverse",
                "rotate",
                "rotate_copy",
                "search",
                "search_n",
                "set_difference",
                "set_intersection",
                "set_symmetric_difference",
                "set_union",
                "sort",
                "stable_partition",
                "unique",
                "unique_copy"]
    }

    for other in vs:
        if not other in competitors:
            continue
        if not name in competitors[other]:
            continue

        for size in sizes:
            time = run_perf_process(name, size, other)
            report.add_sample(other, size, time)

    return report

if __name__ == '__main__':
    test = "sort"
    if len(sys.argv) >= 2:
        test = sys.argv[1]
    print 'running %s perf test' % test

    sizes = [ pow(2, x) for x in range(1, 26) ]

    sizes = sorted(sizes)

    competitors = ["tbb", "thrust", "stl"]

    report = run_benchmark(test, sizes, competitors)

    plot = None
    if "--plot-time" in sys.argv:
        plot = "time"
    elif "--plot-rate" in sys.argv:
        plot = "rate"

    if plot == "time":
        report.plot_time("compute")
        for competitor in competitors:
            report.plot_time(competitor)
    elif plot == "rate":
        report.plot_rate("compute")
        for competitor in competitors:
            report.plot_rate(competitor)

    if plot:
        pylab.legend()
        pylab.show()
    else:
        report.display()
