rounds per cell: [6]

| case | device µs main | device µs fix | device ratio | device | host µs main | host µs fix | host ratio | host |
|---|---|---|---|---|---|---|---|---|
| `elementwise-max-f32-1000` | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | faster (IQRs disjoint) | 4.60 [3.84-5.06] | 3.83 [3.74-4.24] | 0.833 | IQRs overlap |
| `elementwise-max-f64-1000` | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.980 | faster (IQRs disjoint) | 3.90 [3.74-5.16] | 3.77 [3.74-3.79] | 0.965 | IQRs overlap |
| `elementwise-min-f32-1000` | 2.13 [2.13-2.13] | 2.08 [2.08-2.08] | 0.978 | faster (IQRs disjoint) | 3.70 [3.69-3.79] | 3.70 [3.68-4.17] | 1.001 | IQRs overlap |
| `elementwise-min-f64-1000` | 2.18 [2.18-2.18] | 2.14 [2.14-2.14] | 0.981 | faster (IQRs disjoint) | 3.79 [3.75-4.83] | 3.78 [3.71-4.16] | 0.996 | IQRs overlap |
| `elementwise-max-f32-1048576` | 7.79 [7.79-7.80] | 7.63 [7.62-7.63] | 0.979 | faster (IQRs disjoint) | 9.32 [9.20-9.36] | 9.20 [9.10-9.41] | 0.986 | IQRs overlap |
| `elementwise-max-f64-1048576` | 9.32 [9.31-9.33] | 9.25 [9.24-9.25] | 0.992 | faster (IQRs disjoint) | 10.75 [10.75-10.77] | 10.71 [10.69-10.78] | 0.996 | IQRs overlap |
| `elementwise-min-f32-1048576` | 7.80 [7.79-7.80] | 7.63 [7.62-7.63] | 0.978 | faster (IQRs disjoint) | 9.32 [9.18-9.38] | 9.01 [9.01-9.08] | 0.967 | faster (IQRs disjoint) |
| `elementwise-min-f64-1048576` | 9.33 [9.31-9.34] | 9.24 [9.24-9.24] | 0.991 | faster (IQRs disjoint) | 10.82 [10.71-10.93] | 10.69 [10.66-10.71] | 0.988 | faster (IQRs disjoint) |
| `elementwise-max-f32-16777216` | 139.19 [139.14-139.22] | 138.71 [138.70-138.75] | 0.997 | faster (IQRs disjoint) | 140.61 [140.59-140.70] | 140.17 [140.15-140.23] | 0.997 | faster (IQRs disjoint) |
| `elementwise-max-f64-16777216` | 251.18 [251.13-251.20] | 250.88 [250.86-250.96] | 0.999 | faster (IQRs disjoint) | 252.66 [252.56-252.72] | 252.34 [252.27-252.38] | 0.999 | faster (IQRs disjoint) |
| `elementwise-min-f32-16777216` | 139.20 [139.15-139.21] | 138.74 [138.71-138.80] | 0.997 | faster (IQRs disjoint) | 140.65 [140.61-140.65] | 140.20 [140.15-140.26] | 0.997 | faster (IQRs disjoint) |
| `elementwise-min-f64-16777216` | 251.13 [251.08-251.18] | 250.92 [250.91-250.93] | 0.999 | faster (IQRs disjoint) | 252.54 [252.52-252.59] | 252.41 [252.40-252.43] | 0.999 | faster (IQRs disjoint) |

median [Q1-Q3] over the rounds; ratio = fix median / main median.

per-round device µs (main / fix):
- `elementwise-max-f32-1000`: main 2.13 2.13 2.12 2.13 2.13 2.13; fix 2.08 2.08 2.08 2.08 2.08 2.08
- `elementwise-max-f64-1000`: main 2.18 2.18 2.18 2.18 2.18 2.18; fix 2.14 2.14 2.14 2.14 2.14 2.14
- `elementwise-min-f32-1000`: main 2.13 2.13 2.13 2.13 2.13 2.13; fix 2.08 2.08 2.08 2.08 2.08 2.08
- `elementwise-min-f64-1000`: main 2.18 2.18 2.18 2.18 2.18 2.18; fix 2.14 2.13 2.14 2.14 2.14 2.14
- `elementwise-max-f32-1048576`: main 7.80 7.81 7.81 7.78 7.79 7.79; fix 7.63 7.64 7.64 7.62 7.62 7.62
- `elementwise-max-f64-1048576`: main 9.32 9.36 9.33 9.31 9.29 9.33; fix 9.26 9.24 9.26 9.24 9.25 9.24
- `elementwise-min-f32-1048576`: main 7.81 7.80 7.80 7.79 7.79 7.79; fix 7.64 7.63 7.63 7.61 7.62 7.62
- `elementwise-min-f64-1048576`: main 9.33 9.32 9.34 9.34 9.30 9.31; fix 9.25 9.24 9.24 9.25 9.21 9.24
- `elementwise-max-f32-16777216`: main 139.22 139.16 139.29 139.03 139.13 139.21; fix 138.71 138.70 138.76 138.67 138.72 138.78
- `elementwise-max-f64-16777216`: main 251.19 251.21 251.16 251.11 251.02 251.20; fix 250.91 250.99 250.86 250.86 250.98 250.83
- `elementwise-min-f32-16777216`: main 139.13 139.20 139.08 139.37 139.20 139.21; fix 138.66 138.71 138.83 138.81 138.76 138.71
- `elementwise-min-f64-16777216`: main 251.17 251.08 251.03 251.18 251.18 251.09; fix 250.96 250.92 250.90 250.91 250.93 250.91
