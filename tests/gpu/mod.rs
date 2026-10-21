//! GPU test harness: nvcc-compiled cubins, the raw CUDA driver API and
//! by-value argument marshalling. Deliberately independent of
//! `t4a-cuda-runtime`.
//!
//! Both GPU test binaries include this module and use different subsets.
#![allow(dead_code)]

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::result as drv;
use cudarc::driver::sys;
use t4a_cuda_kernels::{CAxisValues, CLayout, CTransposeConfig, Launch};

thread_local! {
    static LAUNCHES: Cell<usize> = const { Cell::new(0) };
}

/// Physical kernel launches performed by the calling test thread, so parallel
/// tests do not observe each other's counts.
pub fn launches() -> usize {
    LAUNCHES.with(Cell::get)
}

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cubin(source: &str) -> PathBuf {
    static CUBINS: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();
    let cache = CUBINS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(path) = cache.get(source) {
        return path.clone();
    }
    let path = {
        let arch = std::env::var("T4A_CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
        let out = std::env::temp_dir().join(format!(
            "t4a-cuda-kernels-{}-{source}.cubin",
            std::process::id()
        ));
        let status = Command::new("nvcc")
            .arg("-std=c++17")
            .arg("-cubin")
            .arg(format!("-arch={arch}"))
            .arg("-I")
            .arg(repo_root().join("cuda/include"))
            .arg("-o")
            .arg(&out)
            .arg(repo_root().join(format!("cuda/src/{source}.cu")))
            .status()
            .expect("nvcc not found: the GPU tests need the CUDA toolkit");
        assert!(status.success(), "nvcc failed for cuda/src/{source}.cu");
        out
    };
    cache.insert(source.to_string(), path.clone());
    path
}

/// One CUDA context, stream and loaded module, owned by the test.
pub struct Session {
    device: sys::CUdevice,
    module: sys::CUmodule,
    stream: sys::CUstream,
}

impl Session {
    pub fn new(source: &str) -> Session {
        drv::init().expect("cuInit failed");
        let device = drv::device::get(0).expect("no usable CUDA device");
        let context =
            unsafe { drv::primary_ctx::retain(device) }.expect("cuDevicePrimaryCtxRetain");
        unsafe { drv::ctx::set_current(context) }.expect("cuCtxSetCurrent");
        let stream =
            drv::stream::create(drv::stream::StreamKind::NonBlocking).expect("cuStreamCreate");
        let image = std::fs::read(cubin(source)).expect("cubin");
        let module =
            unsafe { drv::module::load_data(image.as_ptr().cast()) }.expect("cuModuleLoadData");
        Session {
            device,
            module,
            stream,
        }
    }

    pub fn function(&self, name: &str) -> sys::CUfunction {
        let name = CString::new(name).unwrap();
        unsafe { drv::module::get_function(self.module, name) }.expect("cuModuleGetFunction")
    }

    pub fn stream(&self) -> sys::CUstream {
        self.stream
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { drv::stream::synchronize(self.stream) }.unwrap();
        unsafe { drv::module::unload(self.module) }.unwrap();
        unsafe { drv::stream::destroy(self.stream) }.unwrap();
        unsafe { drv::primary_ctx::release(self.device) }.unwrap();
    }
}

/// Device allocation owned by the test; freed on drop.
pub struct Buf {
    ptr: sys::CUdeviceptr,
    bytes: usize,
}

impl Buf {
    fn allocate(bytes: usize) -> Buf {
        let bytes = bytes.max(1);
        let ptr = unsafe { drv::malloc_sync(bytes) }.expect("cuMemAlloc");
        Buf { ptr, bytes }
    }
}

// Every transfer goes through the session's own non-blocking stream. A blocking
// default-stream memset/memcpy would not be ordered against that stream, and a
// concurrent transfer can then overwrite what the kernel wrote.
pub fn written(session: &Session, bytes: &[u8]) -> Buf {
    let buf = Buf::allocate(bytes.len());
    write(session, &buf, bytes);
    buf
}

pub fn write(session: &Session, buf: &Buf, bytes: &[u8]) {
    assert!(bytes.len() <= buf.bytes);
    unsafe { drv::memcpy_htod_async(buf.ptr, bytes, session.stream()) }.expect("cuMemcpyHtoDAsync");
    sync(session);
}

pub fn read_bytes(session: &Session, buf: &Buf) -> Vec<u8> {
    read_prefix(session, buf, buf.bytes)
}

pub fn read_prefix(session: &Session, buf: &Buf, bytes: usize) -> Vec<u8> {
    assert!(bytes <= buf.bytes);
    let mut host = vec![0u8; bytes];
    unsafe { drv::memcpy_dtoh_async(&mut host, buf.ptr, session.stream()) }
        .expect("cuMemcpyDtoHAsync");
    sync(session);
    host
}

fn sync(session: &Session) {
    unsafe { drv::stream::synchronize(session.stream()) }.expect("cuStreamSynchronize");
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { drv::free_sync(self.ptr) }.unwrap();
    }
}

pub fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: `T: Copy` has no padding-free guarantee, but these tests build the
    // byte image from plain scalars and repr(C) pairs only.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

pub fn from_bytes<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let element = std::mem::size_of::<T>();
    assert_eq!(bytes.len() % element, 0);
    let count = bytes.len() / element;
    // Copy into an aligned `Vec<T>` rather than reinterpreting the byte slice,
    // whose alignment is only 1. Every type used here is a plain scalar or a
    // repr(C) pair of scalars, so any bit pattern is a valid value.
    let mut values: Vec<T> = Vec::with_capacity(count);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        values.set_len(count);
    }
    values
}

pub fn upload<T: Copy>(session: &Session, values: &[T]) -> Buf {
    written(session, as_bytes(values))
}

pub fn download<T: Copy>(session: &Session, buf: &Buf, count: usize) -> Vec<T> {
    from_bytes(&read_prefix(session, buf, count * std::mem::size_of::<T>()))
}

/// A 16-byte-aligned launch-argument slot. The driver copies `size_of::<T>()`
/// bytes for argument `i` from the pointer handed to `cuLaunchKernel`, so every
/// by-value argument (including the 144-byte layout) fits here.
#[repr(C, align(16))]
pub struct Slot([u8; 160]);

impl Slot {
    fn store<T: Copy>(value: &T) -> Slot {
        assert!(std::mem::size_of::<T>() <= 160);
        let mut slot = Slot([0u8; 160]);
        slot.0[..std::mem::size_of::<T>()].copy_from_slice(as_bytes(std::slice::from_ref(value)));
        slot
    }

    pub fn i32(value: i32) -> Slot {
        Slot::store(&value)
    }

    pub fn i64(value: i64) -> Slot {
        Slot::store(&value)
    }

    pub fn layout(value: &CLayout) -> Slot {
        Slot::store(value)
    }

    pub fn config(value: &CTransposeConfig) -> Slot {
        Slot::store(value)
    }

    pub fn axis_values(value: &CAxisValues) -> Slot {
        Slot::store(value)
    }

    pub fn f64(value: f64) -> Slot {
        Slot::store(&value)
    }

    pub fn ptr(value: &Buf) -> Slot {
        Slot::store(&value.ptr)
    }
}

/// Launch and wait. A validated empty plan is skipped, matching the ABI
/// contract, and counts as zero physical launches.
pub fn launch(function: sys::CUfunction, plan: &Launch, stream: sys::CUstream, slots: &mut [Slot]) {
    if plan.is_empty() {
        return;
    }
    let mut params: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|slot| slot.0.as_mut_ptr().cast::<c_void>())
        .collect();
    let grid = (plan.grid[0], plan.grid[1], plan.grid[2]);
    let block = (plan.block[0], plan.block[1], plan.block[2]);
    unsafe {
        drv::launch_kernel(
            function,
            grid,
            block,
            plan.shared_bytes,
            stream,
            &mut params,
        )
    }
    .expect("cuLaunchKernel");
    LAUNCHES.with(|count| count.set(count.get() + 1));
    unsafe { drv::stream::synchronize(stream) }.expect("cuStreamSynchronize");
}
