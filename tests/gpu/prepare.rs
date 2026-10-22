//! Minimal NVRTC prepare / cache / bind consumer for the emitted fused entry.
//!
//! This module is test-only: `examples/bench_kernels.rs` includes it through
//! `#[path]` for its `prepare-fused-cold-compile` case. It re-implements the
//! prepare boundary an external consumer owns and the crate deliberately does
//! not: [`t4a_cuda_kernels::emit_fusion`] produces source, manifest and
//! [`ArtifactKey`]; this module turns that source into a loaded module with
//! NVRTC (`compile_ptx_with_opts`), binds the entry once and caches both under
//! the artifact key that includes the compiler identity, options and target.
//!
//! Counters make every observable step of the contract checkable: emits,
//! NVRTC compile attempts/successes, module loads, cache hits/misses and
//! physical launches. A warm run performs no emit, no NVRTC call and no load.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use cudarc::driver::result as drv;
use cudarc::driver::sys;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use t4a_cuda_kernels::{artifact_key, emit_fusion, ArtifactKey, EmittedFusion, FusionPlan, Launch};

use crate::gpu::{launch_async, Session, Slot};

/// Failure of one prepare/bind/launch step, with the failing stage name.
#[derive(Clone, Debug)]
pub enum PrepareError {
    /// NVRTC refused the emitted source.
    Nvrtc(String),
    /// A CUDA driver call failed with this code.
    Driver { stage: &'static str, code: i32 },
    /// A Rust string contained an interior NUL.
    NullByte(&'static str),
    /// The emitter refused the plan.
    Fusion(t4a_cuda_kernels::Error),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::Nvrtc(log) => write!(f, "NVRTC rejected the emitted source: {log}"),
            PrepareError::Driver { stage, code } => write!(f, "{stage} failed ({code})"),
            PrepareError::NullByte(what) => write!(f, "{what} contains a NUL byte"),
            PrepareError::Fusion(error) => write!(f, "the emitter refused the plan: {error}"),
        }
    }
}

impl std::error::Error for PrepareError {}

/// Observable steps of the consumer; each is independent, so a warm path can
/// assert "no emits and no NVRTC calls".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Calls into [`emit_fusion`].
    pub emits: usize,
    /// NVRTC compile attempts (successful or not).
    pub compile_attempts: usize,
    /// NVRTC compiles that succeeded.
    pub compile_successes: usize,
    /// `cuModuleLoadData` calls.
    pub module_loads: usize,
    /// Cache hits (no compile, no load).
    pub cache_hits: usize,
    /// Cache misses (compile and load).
    pub cache_misses: usize,
    /// Physical kernel launches (a validated empty plan is not one).
    pub launches: usize,
}

/// Compile and load time of the last cache miss, for the cold-path report.
#[derive(Clone, Copy, Debug)]
pub struct MissTimings {
    pub compile: Duration,
    pub load: Duration,
}

/// The architecture this consumer compiles for, from `T4A_CUDA_ARCH` and
/// defaulting to the nvcc harness's `sm_80`.
///
/// `CompileOptions::arch` is a `&'static str`, so the value is interned once.
pub fn target_arch() -> &'static str {
    static ARCH: OnceLock<&'static str> = OnceLock::new();
    ARCH.get_or_init(|| {
        let arch = std::env::var("T4A_CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
        Box::leak(arch.into_boxed_str())
    })
}

/// NVRTC options: the architecture and C++17, and nothing that changes
/// numerical behavior (no `--use_fast_math`, no flush-to-zero).
fn options(arch: &'static str) -> CompileOptions {
    CompileOptions {
        arch: Some(arch),
        options: vec!["-std=c++17".to_string()],
        ..Default::default()
    }
}

/// The canonical option string the artifact key covers. Every
/// [`CompileOptions`] field that can change the generated code appears here, so
/// two different option sets cannot share an artifact.
fn option_key(opts: &CompileOptions) -> String {
    format!(
        "ftz={:?};prec-sqrt={:?};prec-div={:?};fmad={:?};fast-math={:?};maxrregcount={:?};\
         arch={:?};extra={:?}",
        opts.ftz,
        opts.prec_sqrt,
        opts.prec_div,
        opts.fmad,
        opts.use_fast_math,
        opts.maxrregcount,
        opts.arch,
        opts.options,
    )
}

/// The NVRTC version as the compiler-version component of an artifact key.
fn nvrtc_identity() -> String {
    let (mut major, mut minor) = (0i32, 0i32);
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .expect("nvrtcVersion");
    format!("nvrtc {major}.{minor}")
}

/// A prepared artifact: one loaded module and its bound entry.
pub struct Prepared {
    function: sys::CUfunction,
    stream: sys::CUstream,
    counters: Rc<RefCell<Counters>>,
}

impl Prepared {
    /// Launch the prepared entry with the plan's geometry and by-value
    /// arguments. A validated empty plan performs no launch.
    pub fn launch(&self, plan: &Launch, slots: &mut [Slot]) -> Result<(), PrepareError> {
        launch_async(self.function, plan, self.stream, slots);
        self.counters.borrow_mut().launches += 1;
        Ok(())
    }
}

/// A per-session preparation consumer with its own module cache keyed by the
/// artifact key (which already covers source, ABI revision, options, target and
/// compiler identity).
pub struct Consumer<'a> {
    session: &'a Session,
    arch: &'static str,
    target: String,
    compiler: String,
    options: CompileOptions,
    option_key: String,
    modules: RefCell<Vec<sys::CUmodule>>,
    cache: RefCell<HashMap<ArtifactKey, sys::CUfunction>>,
    counters: Rc<RefCell<Counters>>,
    last_miss: RefCell<Option<MissTimings>>,
}

impl<'a> Consumer<'a> {
    /// A consumer on `session`'s context and stream, compiled for
    /// [`target_arch`].
    pub fn new(session: &'a Session) -> Consumer<'a> {
        let arch = target_arch();
        Consumer {
            session,
            target: format!("nvrtc|ptx|{arch}"),
            compiler: nvrtc_identity(),
            options: options(arch),
            option_key: option_key(&options(arch)),
            arch,
            modules: RefCell::new(Vec::new()),
            cache: RefCell::new(HashMap::new()),
            counters: Rc::new(RefCell::new(Counters::default())),
            last_miss: RefCell::new(None),
        }
    }

    pub fn counters(&self) -> Counters {
        *self.counters.borrow()
    }

    pub fn last_miss_timings(&self) -> Option<MissTimings> {
        *self.last_miss.borrow()
    }

    pub fn arch(&self) -> &'static str {
        self.arch
    }

    /// Emit, key and prepare one fused plan: one emit, then a cache hit or one
    /// NVRTC compile plus one module load.
    pub fn prepare_fused(
        &self,
        plan: &FusionPlan,
    ) -> Result<(EmittedFusion, Prepared), PrepareError> {
        self.counters.borrow_mut().emits += 1;
        let emitted = emit_fusion(plan).map_err(PrepareError::Fusion)?;
        let key = artifact_key(&emitted, &self.option_key, &self.target, &self.compiler);
        let prepared = self.prepare(&emitted, &key)?;
        Ok((emitted, prepared))
    }

    /// Prepare one artifact key: cache hit, or compile + load + bind + insert.
    fn prepare(
        &self,
        emitted: &EmittedFusion,
        key: &ArtifactKey,
    ) -> Result<Prepared, PrepareError> {
        if let Some(function) = self.cache.borrow().get(key) {
            self.counters.borrow_mut().cache_hits += 1;
            return Ok(Prepared {
                function: *function,
                stream: self.session.stream(),
                counters: Rc::clone(&self.counters),
            });
        }

        self.counters.borrow_mut().cache_misses += 1;
        self.counters.borrow_mut().compile_attempts += 1;
        let compile_start = Instant::now();
        let image = compile_ptx_with_opts(emitted.source(), self.options.clone())
            .map_err(|error| PrepareError::Nvrtc(format!("{error:?}")))?
            .as_bytes()
            .expect("NVRTC returned a PTX image")
            .to_vec();
        let compile_time = compile_start.elapsed();
        self.counters.borrow_mut().compile_successes += 1;

        let load_start = Instant::now();
        let module = unsafe { drv::module::load_data(image.as_ptr().cast()) }.map_err(|error| {
            PrepareError::Driver {
                stage: "cuModuleLoadData",
                code: error.0 as i32,
            }
        })?;
        let symbol = std::ffi::CString::new(emitted.manifest().entry)
            .map_err(|_| PrepareError::NullByte("entry"))?;
        let function = unsafe { drv::module::get_function(module, symbol) }.map_err(|error| {
            PrepareError::Driver {
                stage: "cuModuleGetFunction",
                code: error.0 as i32,
            }
        })?;
        *self.last_miss.borrow_mut() = Some(MissTimings {
            compile: compile_time,
            load: load_start.elapsed(),
        });
        self.counters.borrow_mut().module_loads += 1;
        self.modules.borrow_mut().push(module);
        self.cache.borrow_mut().insert(key.clone(), function);
        Ok(Prepared {
            function,
            stream: self.session.stream(),
            counters: Rc::clone(&self.counters),
        })
    }
}

impl Drop for Consumer<'_> {
    fn drop(&mut self) {
        unsafe { drv::stream::synchronize(self.session.stream()) }.expect("cuStreamSynchronize");
        for module in self.modules.borrow_mut().drain(..) {
            unsafe { drv::module::unload(module) }.expect("cuModuleUnload");
        }
    }
}
