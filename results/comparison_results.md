# Baseline vs Multi-Agent Comparison Results

> **Note:** These results were produced with the original Llama 3.1/3.3 models via
> Groq, prior to the migration to GPT-OSS-20B/120B following Groq's model
> deprecation announcement. See `README.md`.

## Per-question results

| ID | Category | Pipeline | Faithfulness | Correct | Latency (s) |
|---|---|---|---|---|---|
| sh_001 | single_hop | baseline | 1.00 | True | 8.3 |
| sh_001 | single_hop | multi_agent | 0.50 | False | 2.4 |
| sh_002 | single_hop | baseline | 1.00 | False | 10.1 |
| sh_002 | single_hop | multi_agent | 0.50 | True | 44.6 |
| sh_003 | single_hop | baseline | 0.67 | False | 6.2 |
| sh_003 | single_hop | multi_agent | 0.83 | False | 42.7 |
| sh_004 | single_hop | baseline | 0.00 | False | 0.6 |
| sh_004 | single_hop | multi_agent | 0.00 | False | 36.4 |
| sh_005 | single_hop | baseline | 1.00 | True | 0.5 |
| sh_005 | single_hop | multi_agent | 1.00 | True | 5.6 |
| sh_006 | single_hop | baseline | 0.00 | False | 0.6 |
| sh_006 | single_hop | multi_agent | 0.00 | False | 1.3 |
| sh_007 | single_hop | baseline | 0.00 | False | 10.3 |
| sh_007 | single_hop | multi_agent | 0.17 | False | 34.6 |
| sh_008 | single_hop | baseline | 1.00 | True | 13.9 |
| sh_008 | single_hop | multi_agent | 0.00 | False | 17.2 |
| sh_009 | single_hop | baseline | 1.00 | False | 13.0 |
| sh_009 | single_hop | multi_agent | 0.67 | False | 30.9 |
| sh_010 | single_hop | baseline | 0.75 | True | 12.2 |
| sh_010 | single_hop | multi_agent | 1.00 | True | 17.3 |
| i2h_011 | independent_2hop | baseline | 1.00 | False | 6.5 |
| i2h_011 | independent_2hop | multi_agent | 0.88 | False | 34.3 |
| i2h_012 | independent_2hop | baseline | 1.00 | False | 0.9 |
| i2h_012 | independent_2hop | multi_agent | 0.83 | False | 56.9 |
| i2h_013 | independent_2hop | baseline | 1.00 | False | 10.3 |
| i2h_013 | independent_2hop | multi_agent | 1.00 | False | 3.3 |
| i2h_014 | independent_2hop | baseline | 1.00 | False | 7.2 |
| i2h_014 | independent_2hop | multi_agent | 0.88 | False | 58.7 |
| i2h_015 | independent_2hop | baseline | 1.00 | False | 1.1 |
| i2h_015 | independent_2hop | multi_agent | 0.67 | False | 81.2 |
| i2h_016 | independent_2hop | baseline | 1.00 | False | 1.0 |
| i2h_016 | independent_2hop | multi_agent | 0.93 | False | 88.9 |
| i2h_017 | independent_2hop | baseline | 1.00 | True | 0.8 |
| i2h_017 | independent_2hop | multi_agent | 0.83 | False | 20.1 |
| i2h_018 | independent_2hop | baseline | 0.67 | False | 1.1 |
| i2h_018 | independent_2hop | multi_agent | 0.83 | False | 74.5 |
| i2h_019 | independent_2hop | baseline | 0.83 | False | 0.8 |
| i2h_019 | independent_2hop | multi_agent | 1.00 | False | 53.7 |
| i2h_020 | independent_2hop | baseline | 0.67 | False | 1.3 |
| i2h_020 | independent_2hop | multi_agent | 0.83 | False | 20.8 |
| i2h_021 | independent_2hop | baseline | 0.50 | False | 1.7 |
| i2h_021 | independent_2hop | multi_agent | 0.86 | False | 115.9 |
| i2h_022 | independent_2hop | baseline | 0.75 | True | 4.3 |
| i2h_022 | independent_2hop | multi_agent | 0.77 | False | 68.3 |
| d2h_023 | dependent_2hop | baseline | 1.00 | False | 0.9 |
| d2h_023 | dependent_2hop | multi_agent | 0.54 | False | 102.1 |
| d2h_024 | dependent_2hop | baseline | 0.67 | False | 0.7 |
| d2h_024 | dependent_2hop | multi_agent | 0.40 | False | 78.7 |

## Aggregate results by category

| Category | Pipeline | Avg Faithfulness | Accuracy | Avg Latency (s) | N |
|---|---|---|---|---|---|
| single_hop | baseline | 0.64 | 0.40 | 7.6 | 10 |
| single_hop | multi_agent | 0.47 | 0.30 | 23.3 | 10 |
| independent_2hop | baseline | 0.87 | 0.17 | 3.1 | 12 |
| independent_2hop | multi_agent | 0.86 | 0.00 | 56.4 | 12 |
| dependent_2hop | baseline | 0.83 | 0.00 | 0.8 | 2 |
| dependent_2hop | multi_agent | 0.47 | 0.00 | 90.4 | 2 |

## Overall

| Pipeline | Avg Faithfulness | Accuracy | Avg Latency (s) | N |
|---|---|---|---|---|
| baseline | 0.77 | 0.25 | 4.8 | 24 |
| multi_agent | 0.66 | 0.12 | 45.4 | 24 |
