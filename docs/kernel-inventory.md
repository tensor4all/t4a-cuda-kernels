# CubeCL kernel migration inventory

Source: `tensor4all/tenferro-rs@14bc1ab933e3c7a9c966e08355670331ab9d993c`. This is an inventory of source entry points and device helpers, not a claim of implementation or numerical validation. Generic dtype instantiations are not counted separately. Macro-generated symbols are listed explicitly where identifiable.

The design and migration decisions are in [the design note](design/cubecl-kernel-port.md).

## crates/tenferro-gpu/src/kernels/elementwise.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs); 882 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `broadcast_source_index` | helper | [L58](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L58) |
| `broadcast_multiply_float` | macro_entry | [L103](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L103) |
| `broadcast_multiply_complex` | macro_entry | [L104](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L104) |
| `broadcast_multiply_int` | entry | [L106](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L106) |
| `axpby_strided_source_float` | macro_entry | [L200](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L200) |
| `axpby_strided_source_complex` | macro_entry | [L201](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L201) |
| `add_complex` | macro_entry | [L203](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L203) |
| `add_float` | macro_entry | [L203](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L203) |
| `sub_complex` | macro_entry | [L204](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L204) |
| `sub_float` | macro_entry | [L204](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L204) |
| `mul_complex` | macro_entry | [L205](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L205) |
| `mul_float` | macro_entry | [L205](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L205) |
| `div_float` | macro_entry | [L208](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L208) |
| `add_int` | entry | [L210](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L210) |
| `sub_int` | entry | [L217](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L217) |
| `mul_int` | entry | [L224](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L224) |
| `scalar_div_float` | macro_entry | [L254](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L254) |
| `scalar_pow_float` | macro_entry | [L255](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L255) |
| `scalar_rem_float` | macro_entry | [L262](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L262) |
| `scalar_real_complex_binary` | entry | [L264](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L264) |
| `scalar_div_int_checked` | entry | [L332](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L332) |
| `scalar_rem_int_checked` | entry | [L366](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L366) |
| `div_int_checked` | entry | [L401](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L401) |
| `rem_float` | entry | [L419](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L419) |
| `rem_int_checked` | entry | [L428](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L428) |
| `neg_complex` | macro_entry | [L446](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L446) |
| `neg_float` | macro_entry | [L446](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L446) |
| `neg_int` | entry | [L447](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L447) |
| `exp_float` | macro_entry | [L453](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L453) |
| `log_float` | macro_entry | [L454](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L454) |
| `sin_float` | macro_entry | [L455](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L455) |
| `cos_float` | macro_entry | [L456](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L456) |
| `tanh_float` | macro_entry | [L457](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L457) |
| `sqrt_float` | macro_entry | [L458](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L458) |
| `rsqrt_float` | entry | [L460](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L460) |
| `expm1_float` | entry | [L467](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L467) |
| `log1p_float` | entry | [L474](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L474) |
| `erf_float` | entry | [L481](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L481) |
| `abs_float` | entry | [L488](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L488) |
| `abs_complex32` | entry | [L496](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L496) |
| `abs_complex64` | entry | [L503](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L503) |
| `abs_int` | entry | [L510](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L510) |
| `sign_float` | entry | [L523](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L523) |
| `sign_complex` | entry | [L541](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L541) |
| `sign_int` | entry | [L554](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L554) |
| `maximum_float` | entry | [L569](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L569) |
| `maximum_int` | entry | [L576](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L576) |
| `minimum_float` | entry | [L585](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L585) |
| `minimum_int` | entry | [L592](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L592) |
| `pow_float` | entry | [L601](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L601) |
| `pow_int_checked` | entry | [L608](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L608) |
| `scalar_pow_int_checked` | entry | [L637](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L637) |
| `compare_float_bool` | entry | [L682](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L682) |
| `compare_int_bool` | entry | [L704](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L704) |
| `select_bool_float` | entry | [L721](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L721) |
| `select_bool_int` | entry | [L732](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L732) |
| `clamp_float` | entry | [L743](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L743) |
| `conj_complex` | entry | [L750](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L750) |
| `complex_div_abs` | helper | [L759](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L759) |
| `complex_div_sign` | helper | [L769](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L769) |
| `complex_div_term` | helper | [L782](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L782) |
| `div_complex_parts` | entry | [L803](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L803) |
| `complex_div_parts_at` | helper | [L819](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/elementwise.rs#L819) |

## crates/tenferro-gpu/src/kernels/structural.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs); 589 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `strided_view_offset_from_tensor` | helper | [L8](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L8) |
| `fill_zero_kernel` | entry | [L27](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L27) |
| `fill_zero_view_kernel` | entry | [L40](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L40) |
| `copy_bool_kernel` | entry | [L63](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L63) |
| `scale_in_place_float_kernel` | entry | [L73](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L73) |
| `scale_in_place_complex_kernel` | entry | [L91](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L91) |
| `materialize_strided_kernel` | entry | [L102](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L102) |
| `materialize_strided_comptime_layout_kernel` | entry | [L133](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L133) |
| `tiled_transpose_kernel` | entry | [L164](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L164) |
| `tiled_transpose_comptime_layout_kernel` | entry | [L227](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L227) |
| `contiguous_to_view_kernel` | entry | [L284](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L284) |
| `strided_to_strided_kernel` | entry | [L299](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L299) |
| `broadcast_in_dim_kernel` | entry | [L327](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L327) |
| `convert_float_to_float` | entry | [L351](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L351) |
| `convert_numeric` | entry | [L358](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L358) |
| `convert_numeric_to_bool` | entry | [L365](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L365) |
| `convert_bool_to_numeric` | entry | [L376](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L376) |
| `convert_numeric_to_complex_raw` | entry | [L383](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L383) |
| `convert_bool_to_complex_raw` | entry | [L391](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L391) |
| `convert_complex_to_numeric` | entry | [L399](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L399) |
| `convert_complex_raw_to_bool` | entry | [L406](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L406) |
| `validate_real_cast` | entry | [L419](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L419) |
| `extract_invalid_real_cast` | entry | [L447](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L447) |
| `convert_c32_to_f32` | entry | [L462](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L462) |
| `convert_c32_to_f64` | entry | [L469](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L469) |
| `convert_c64_to_f32` | entry | [L476](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L476) |
| `convert_c64_to_f64` | entry | [L483](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L483) |
| `convert_f32_to_c32_raw` | entry | [L494](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L494) |
| `convert_f32_to_c64_raw` | entry | [L503](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L503) |
| `convert_f64_to_c32_raw` | entry | [L512](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L512) |
| `convert_f64_to_c64_raw` | entry | [L521](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L521) |
| `convert_complex_to_complex` | entry | [L530](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L530) |
| `convert_complex_raw` | entry | [L540](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L540) |
| `reverse_kernel` | entry | [L547](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L547) |
| `concatenate_copy_kernel` | entry | [L570](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/structural.rs#L570) |

## crates/tenferro-gpu/src/kernels/indexing.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs); 558 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `init_float_index_validation_flag` | entry | [L12](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L12) |
| `validate_float_indices_kernel` | entry | [L22](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L22) |
| `extract_invalid_float_index_kernel` | entry | [L47](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L47) |
| `clamp_window_start` | helper | [L63](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L63) |
| `index_component` | helper | [L82](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L82) |
| `flat_to_index_batch_index` | helper | [L108](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L108) |
| `flat_to_update_window_index` | helper | [L139](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L139) |
| `update_window_len` | helper | [L160](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L160) |
| `slice_kernel` | entry | [L174](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L174) |
| `dynamic_slice_kernel` | entry | [L193](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L193) |
| `pad_kernel` | entry | [L215](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L215) |
| `gather_kernel` | entry | [L263](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L263) |
| `scatter_copy_kernel` | entry | [L343](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L343) |
| `scatter_float_kernel` | entry | [L350](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L350) |
| `scatter_complex_kernel` | entry | [L455](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/indexing.rs#L455) |

## crates/tenferro-gpu/src/kernels/diagonal.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs); 87 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `extract_diagonal_kernel` | entry | [L5](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs#L5) |
| `embed_diagonal_copy_kernel` | entry | [L33](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs#L33) |
| `tril_kernel` | entry | [L61](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs#L61) |
| `triu_kernel` | entry | [L75](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/diagonal.rs#L75) |

## crates/tenferro-gpu/src/kernels/reduce/kernels.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs); 1438 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `reduce_sum_float` | macro_entry | [L60](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L60) |
| `reduce_sum_complex` | macro_entry | [L61](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L61) |
| `reduce_prod_float` | macro_entry | [L62](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L62) |
| `reduce_prod_complex` | macro_entry | [L63](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L63) |
| `reduce_sum_squares_float` | entry | [L65](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L65) |
| `reduce_sum_int` | entry | [L105](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L105) |
| `reduce_prod_int` | entry | [L140](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L140) |
| `reduce_sum_float_plane` | entry | [L175](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L175) |
| `reduce_sum_float_chunked` | entry | [L223](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L223) |
| `reduce_sum_complex_chunked` | entry | [L282](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L282) |
| `reduce_prod_float_chunked` | entry | [L338](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L338) |
| `reduce_prod_complex_chunked` | entry | [L394](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L394) |
| `reduce_sum_squares_float_chunked` | entry | [L450](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L450) |
| `reduce_sum_int_chunked` | entry | [L511](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L511) |
| `reduce_prod_int_chunked` | entry | [L567](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L567) |
| `reduce_max_float_chunked` | entry | [L623](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L623) |
| `reduce_min_float_chunked` | entry | [L685](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L685) |
| `reduce_max_int_chunked` | entry | [L747](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L747) |
| `reduce_min_int_chunked` | entry | [L804](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L804) |
| `reduce_sum_complex_plane` | entry | [L861](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L861) |
| `reduce_prod_float_plane` | entry | [L903](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L903) |
| `reduce_prod_complex_plane` | entry | [L945](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L945) |
| `reduce_sum_squares_float_plane` | entry | [L987](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L987) |
| `reduce_sum_int_plane` | entry | [L1032](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1032) |
| `reduce_prod_int_plane` | entry | [L1074](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1074) |
| `reduce_max_float` | entry | [L1116](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1116) |
| `reduce_max_int` | entry | [L1151](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1151) |
| `reduce_max_float_plane` | entry | [L1187](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1187) |
| `reduce_max_int_plane` | entry | [L1235](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1235) |
| `reduce_min_float` | entry | [L1278](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1278) |
| `reduce_min_int` | entry | [L1313](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1313) |
| `reduce_min_float_plane` | entry | [L1349](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1349) |
| `reduce_min_int_plane` | entry | [L1397](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce/kernels.rs#L1397) |

## crates/tenferro-gpu/src/kernels/helpers.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs); 140 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `wrapping_add` | helper | [L7](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L7) |
| `wrapping_sub` | helper | [L12](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L12) |
| `wrapping_mul` | helper | [L17](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L17) |
| `wrapping_neg` | helper | [L22](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L22) |
| `wrapping_plane_sum` | helper | [L27](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L27) |
| `wrapping_plane_prod` | helper | [L32](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L32) |
| `nan_propagating_max` | helper | [L40](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L40) |
| `nan_propagating_min` | helper | [L52](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L52) |
| `plane_contains_nan` | helper | [L64](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L64) |
| `plane_propagate_nan` | helper | [L73](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L73) |
| `zero_value` | helper | [L83](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L83) |
| `flat_to_tensor_index` | helper | [L88](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L88) |
| `multi_to_tensor_index` | helper | [L102](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L102) |
| `axis_in_sequence` | helper | [L116](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L116) |
| `axis_position_in_sequence` | helper | [L129](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/helpers.rs#L129) |

## crates/tenferro-linalg/src/gpu/kernels.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs); 491 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `zero_value` | helper | [L8](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L8) |
| `one_value` | helper | [L13](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L13) |
| `batch_linear_index` | helper | [L18](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L18) |
| `matching_work_offset` | helper | [L36](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L36) |
| `matrix_adjoint_offset` | helper | [L54](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L54) |
| `fill_one_kernel` | entry | [L72](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L72) |
| `matrix_adjoint_real` | entry | [L80](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L80) |
| `matrix_adjoint_complex` | entry | [L92](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L92) |
| `complex32_magnitude` | entry | [L104](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L104) |
| `complex64_magnitude` | entry | [L111](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L111) |
| `householder_explicit_v` | entry | [L118](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L118) |
| `householder_q_columns_identity` | entry | [L134](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L134) |
| `positive_phase_real` | helper | [L151](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L151) |
| `positive_phase_c32` | helper | [L164](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L164) |
| `positive_phase_c64` | helper | [L174](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L174) |
| `qr_diagonal_offset` | helper | [L184](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L184) |
| `qr_phase_offset` | helper | [L200](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L200) |
| `qr_phase_real` | entry | [L216](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L216) |
| `qr_phase_c32` | entry | [L232](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L232) |
| `qr_phase_c64` | entry | [L242](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L242) |
| `qr_apply_phase_real` | entry | [L252](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L252) |
| `qr_apply_phase_complex` | entry | [L275](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L275) |
| `upper_trapezoidal_violation` | entry | [L297](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L297) |
| `householder_from_factors_assemble` | entry | [L314](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L314) |
| `lu_extract_outputs` | entry | [L344](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L344) |
| `lu_parity` | entry | [L419](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L419) |
| `lu_apply_pivots` | entry | [L441](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels.rs#L441) |

## crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs

[Pinned source](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs); 762 physical lines including comments and blank lines.

| Symbol | Kind | Source |
| --- | --- | --- |
| `rrqr_zero` | helper | [L14](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L14) |
| `rrqr_one` | helper | [L19](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L19) |
| `initialize_permutation` | entry | [L24](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L24) |
| `column_norms_real` | entry | [L32](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L32) |
| `column_norms_c32` | entry | [L79](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L79) |
| `column_norms_c64` | entry | [L128](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L128) |
| `select_pivot` | entry | [L177](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L177) |
| `swap_columns` | entry | [L232](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L232) |
| `reflector_real` | entry | [L265](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L265) |
| `c32_from_real` | helper | [L325](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L325) |
| `c64_from_real` | helper | [L330](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L330) |
| `reflector_c32` | entry | [L335](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L335) |
| `reflector_c64` | entry | [L394](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L394) |
| `apply_reflector_real` | entry | [L453](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L453) |
| `apply_reflector_c32` | entry | [L499](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L499) |
| `apply_reflector_c64` | entry | [L547](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L547) |
| `initialize_q` | entry | [L595](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L595) |
| `extract_r` | entry | [L609](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L609) |
| `prefix_rank_real` | entry | [L630](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L630) |
| `prefix_rank_c32` | entry | [L676](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L676) |
| `prefix_rank_c64` | entry | [L720](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs#L720) |

## Supporting and excluded paths

| Path | Disposition |
| --- | --- |
| [reduce launch/routines](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/kernels/reduce) | Preserve algorithm/strategy/workspace requirements; remove CubeCL client/launch plumbing. |
| [native permutation planner](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/native_permutation.rs) | Host layout/planning reference, not CUDA kernel source. |
| [fusion](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/cubecl/fusion) | Dynamic CubeCL IR generation; tenferro selects fusion groups and adapts them to an independent expression description. The kernel crate emits CUDA with runtime shape arguments; the caller compiles/caches it during prepare. No dependency on tenferro fusion-plan types. |
| [WebGPU packing kernels](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-gpu/src/webgpu/kernels.rs) | WebGPU/CubeK route only; not required for current CUDA parity. |
| [CUDA FFT](https://github.com/tensor4all/tenferro-rs/tree/14bc1ab933e3c7a9c966e08355670331ab9d993c/crates/tenferro-fft/src/cuda) | cuFFT execution/plan integration; no separate static CubeCL FFT kernel definitions found in this source tree. |
| [sample CubeCL kernel](https://github.com/tensor4all/tenferro-rs/blob/14bc1ab933e3c7a9c966e08355670331ab9d993c/samples/cubecl-kernel/src/kernel.rs) | Consumer example, not production parity inventory. |
