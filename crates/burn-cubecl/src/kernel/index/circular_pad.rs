use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};

// ---------------------------------------------------------------------------
// Forward kernel: one thread per output element
//
// Output shape: [N, C, H + 2*pad_h, W + 2*pad_w]
// For each output position (n,c,oh,ow):
//   ih = oh - pad_h
//   iw = (ow - pad_w + W) % W    (circular wrap)
//   if ih < 0 or ih >= H: output 0  (height zero-pad)
//   else: output = input[n,c,ih,iw]
// ---------------------------------------------------------------------------
#[cube(launch_unchecked, address_type = "dynamic")]
fn circular_pad_2d_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    pad_h: usize,
    pad_w: usize,
    orig_h: usize,
    orig_w: usize,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    // Decompose ABSOLUTE_POS into [n, c, oh, ow] using output shape divmods.
    // Process dims in reverse order (ow, oh, c, n).
    let mut rem = ABSOLUTE_POS;

    let (r, ow) = out_shape[3].div_mod(rem);
    rem = r;
    let (r, oh) = out_shape[2].div_mod(rem);
    rem = r;
    let (r, c) = out_shape[1].div_mod(rem);
    rem = r;
    let n = rem;
    let _ = r;

    // Check height bounds (zero-pad region)
    if oh < pad_h || oh >= pad_h + orig_h {
        output[ABSOLUTE_POS] = E::from_int(0);
    } else {
        // Map output coords to input coords
        let ih = oh - pad_h;
        // Circular wrap: iw = (ow - pad_w + orig_w) % orig_w
        let iw = (ow + orig_w - pad_w) % orig_w;

        let input_offset =
            n * input.stride(0) + c * input.stride(1) + ih * input.stride(2) + iw * input.stride(3);
        output[ABSOLUTE_POS] = input[input_offset];
    }
}

// ---------------------------------------------------------------------------
// Backward kernel: one thread per input element (no atomics)
//
// Each input element (n,c,h,w) reads gradients from ALL output positions
// that it was copied to via circular wrapping. These positions are:
//   ow = w + pad_w + k * orig_w   for all integer k where 0 <= ow < out_w
//   ow = w + pad_w - k * orig_w   for all integer k where 0 <= ow < out_w
// Equivalently: all ow where ow ≡ w + pad_w (mod orig_w), 0 <= ow < out_w.
// ---------------------------------------------------------------------------
#[cube(launch_unchecked, address_type = "dynamic")]
fn circular_pad_2d_backward_kernel<E: Numeric>(
    grad_output: &Tensor<E>,
    grad_input: &mut LinearView<E, ReadWrite>,
    in_shape: Sequence<FastDivmod<usize>>,
    pad_h: usize,
    pad_w: usize,
    orig_w: usize,
    out_w: usize,
    #[define(E)] _dtype: StorageType,
) {
    if !grad_input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    // Decompose ABSOLUTE_POS into [n, c, h, w] using input shape divmods.
    let mut rem = ABSOLUTE_POS;

    let (r, w) = in_shape[3].div_mod(rem);
    rem = r;
    let (r, h) = in_shape[2].div_mod(rem);
    rem = r;
    let (r, c) = in_shape[1].div_mod(rem);
    rem = r;
    let n = rem;
    let _ = r;

    let oh = h + pad_h;
    let go_base =
        n * grad_output.stride(0) + c * grad_output.stride(1) + oh * grad_output.stride(2);
    let go_stride_w = grad_output.stride(3);

    // Sum gradients from all periodic copies of this input column.
    // First occurrence is at ow = (w + pad_w) % orig_w, then every +orig_w.
    let first_ow = (w + pad_w) % orig_w;
    let mut g = E::from_int(0);
    let mut ow = first_ow;
    while ow < out_w {
        g += grad_output[go_base + ow * go_stride_w];
        ow += orig_w;
    }

    grad_input[ABSOLUTE_POS] = g;
}

// ---------------------------------------------------------------------------
// Launch functions
// ---------------------------------------------------------------------------

pub(crate) fn circular_pad_2d<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    pad_h: usize,
    pad_w: usize,
) -> CubeTensor<R> {
    let in_shape = tensor.meta.shape();
    let h = in_shape[2];
    let w = in_shape[3];

    let out_shape = Shape::from(vec![in_shape[0], in_shape[1], h + 2 * pad_h, w + 2 * pad_w]);

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        out_shape,
        tensor.dtype,
    );

    let dtype_input = tensor.dtype;
    let num_elements = output.meta.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, num_elements, cube_dim);

    unsafe {
        circular_pad_2d_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            address_type!(tensor, output),
            tensor.as_tensor_arg(1),
            linear_view(&output, 1),
            shape_divmod(&output),
            ScalarArg::new(pad_h),
            ScalarArg::new(pad_w),
            ScalarArg::new(h),
            ScalarArg::new(w),
            dtype_input.into(),
        )
        .expect("circular_pad_2d forward kernel failed");
    }

    output
}

pub(crate) fn circular_pad_2d_backward<R: CubeRuntime>(
    grad_output: CubeTensor<R>,
    pad_h: usize,
    pad_w: usize,
    original_h: usize,
    original_w: usize,
) -> CubeTensor<R> {
    let in_shape = Shape::from(vec![
        grad_output.meta.shape()[0],
        grad_output.meta.shape()[1],
        original_h,
        original_w,
    ]);

    let grad_input = empty_device_dtype(
        grad_output.client.clone(),
        grad_output.device.clone(),
        in_shape,
        grad_output.dtype,
    );

    let out_w = original_w + 2 * pad_w;
    let dtype_input = grad_output.dtype;
    let num_elements = grad_input.meta.num_elements();
    let cube_dim = CubeDim::new(&grad_output.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&grad_output.client, num_elements, cube_dim);

    unsafe {
        circular_pad_2d_backward_kernel::launch_unchecked(
            &grad_output.client,
            cube_count,
            cube_dim,
            address_type!(grad_output, grad_input),
            grad_output.as_tensor_arg(1),
            linear_view(&grad_input, 1),
            shape_divmod(&grad_input),
            ScalarArg::new(pad_h),
            ScalarArg::new(pad_w),
            ScalarArg::new(original_w),
            ScalarArg::new(out_w),
            dtype_input.into(),
        )
        .expect("circular_pad_2d backward kernel failed");
    }

    grad_input
}
