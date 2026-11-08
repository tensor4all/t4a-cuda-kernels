rounds per cell: [4]

| case | device µs main | device µs fix | device ratio | device | host µs main | host µs fix | host ratio | host |
|---|---|---|---|---|---|---|---|---|
| `elementwise-max-f32-1000` | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | faster (IQRs disjoint) | 3.73 [3.72-3.88] | 4.33 [3.76-5.05] | 1.161 | IQRs overlap |
| `elementwise-max-f64-1000` | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.980 | faster (IQRs disjoint) | 3.78 [3.76-4.15] | 4.89 [3.73-6.07] | 1.291 | IQRs overlap |
| `elementwise-min-f32-1000` | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | faster (IQRs disjoint) | 3.76 [3.71-4.33] | 3.71 [3.68-3.75] | 0.986 | IQRs overlap |
| `elementwise-min-f64-1000` | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.981 | faster (IQRs disjoint) | 3.72 [3.66-3.85] | 3.66 [3.60-3.71] | 0.984 | IQRs overlap |
| `elementwise-max-f32-1048576` | 7.80 [7.80-7.80] | 7.64 [7.64-7.64] | 0.979 | faster (IQRs disjoint) | 9.22 [9.19-9.29] | 9.08 [9.07-9.16] | 0.985 | faster (IQRs disjoint) |
| `elementwise-max-f64-1048576` | 9.34 [9.34-9.35] | 9.25 [9.24-9.26] | 0.990 | faster (IQRs disjoint) | 11.01 [10.78-11.52] | 10.65 [10.60-10.70] | 0.967 | faster (IQRs disjoint) |
| `elementwise-min-f32-1048576` | 7.80 [7.80-7.81] | 7.63 [7.63-7.64] | 0.978 | faster (IQRs disjoint) | 9.25 [9.23-9.33] | 9.09 [9.05-9.17] | 0.983 | faster (IQRs disjoint) |
| `elementwise-min-f64-1048576` | 9.33 [9.32-9.35] | 9.25 [9.24-9.25] | 0.991 | faster (IQRs disjoint) | 10.70 [10.68-10.72] | 10.67 [10.65-10.73] | 0.997 | IQRs overlap |
| `elementwise-max-f32-16777216` | 139.24 [139.21-139.27] | 138.71 [138.68-138.74] | 0.996 | faster (IQRs disjoint) | 140.66 [140.64-141.04] | 140.17 [140.13-140.20] | 0.997 | faster (IQRs disjoint) |
| `elementwise-max-f64-16777216` | 251.12 [251.09-251.16] | 250.94 [250.92-250.97] | 0.999 | faster (IQRs disjoint) | 252.54 [252.48-252.68] | 252.45 [252.40-252.55] | 1.000 | IQRs overlap |
| `elementwise-min-f32-16777216` | 139.23 [139.22-139.24] | 138.73 [138.73-138.75] | 0.996 | faster (IQRs disjoint) | 140.65 [140.60-140.69] | 140.18 [140.13-140.24] | 0.997 | faster (IQRs disjoint) |
| `elementwise-min-f64-16777216` | 251.12 [251.09-251.15] | 250.90 [250.89-250.91] | 0.999 | faster (IQRs disjoint) | 252.62 [252.50-253.05] | 252.33 [252.28-252.36] | 0.999 | faster (IQRs disjoint) |

median [Q1-Q3] over the rounds; ratio = fix median / main median.

per-round device µs (main / fix):
- `elementwise-max-f32-1000`: main 2.13 2.13 2.13 2.13; fix 2.08 2.08 2.08 2.08
- `elementwise-max-f64-1000`: main 2.18 2.18 2.18 2.18; fix 2.14 2.14 2.14 2.14
- `elementwise-min-f32-1000`: main 2.13 2.13 2.13 2.13; fix 2.08 2.08 2.08 2.08
- `elementwise-min-f64-1000`: main 2.18 2.18 2.18 2.18; fix 2.14 2.14 2.14 2.13
- `elementwise-max-f32-1048576`: main 7.80 7.81 7.80 7.80; fix 7.64 7.64 7.64 7.63
- `elementwise-max-f64-1048576`: main 9.35 9.33 9.34 9.35; fix 9.24 9.26 9.24 9.26
- `elementwise-min-f32-1048576`: main 7.80 7.80 7.80 7.81; fix 7.63 7.64 7.63 7.63
- `elementwise-min-f64-1048576`: main 9.31 9.35 9.32 9.35; fix 9.27 9.24 9.25 9.23
- `elementwise-max-f32-16777216`: main 139.30 139.18 139.27 139.22; fix 138.68 138.67 138.74 138.74
- `elementwise-max-f64-16777216`: main 251.07 251.19 251.09 251.15; fix 250.93 250.90 250.94 251.03
- `elementwise-min-f32-16777216`: main 139.22 139.22 139.23 139.26; fix 138.74 138.79 138.72 138.73
- `elementwise-min-f64-16777216`: main 251.02 251.11 251.13 251.22; fix 250.90 250.94 250.84 250.91
