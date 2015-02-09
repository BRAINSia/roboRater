roboRater
=========

Automated QA ratings based on previous manual QA and similarity
metrics

```
Usage:
  roboRater -h | --help
  roboRater -V | --version
  roboRater test [-v | --verbose]
  roboRater compare [--force] CONFIG REPORT
  roboRater write   [--force] TEST BENCHMARK RESULTS CSVIN SQLOUT

Commands:
  compare           Run comparison
  test              Run doctests
  write             Run comparison and create merged segmentations based on results

Arguments:
  CONFIG            Configuration file defining all inputs/outputs for roboRater
  BENCHMARK         Benchmark experiment directory, i.e. the experiment to compare to
  TEST              Test experiment directory, i.e. the experiment to compare
  RESULTS           Result experiment directory
  REPORT            Report file path
  CSVIN             Comma-separated file path with headings
  SQLOUT            SQL-generated output file name

Options:
  -h, --help           Show this help and exit
  -V, --version        Print roboRater version
  -v, --verbose        Print more testing information
  --force              Overwrite previous results
```

Examples
--------
Compare results from two experiments and write out SQL query

```bash
    python roboRater.py compare /path/to/test/site/subject/session/result \
      /path/to/benchmark \
      /path/to/result \
      /path/to/report.sql
```
