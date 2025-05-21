mod kernel_fn;
mod module;
mod ptx;

use std::{ffi::CString, str::FromStr};

pub use kernel_fn::{KernelFn, KernelParamPtrs, KernelParams};
pub use module::{Module, ModuleSpore};
pub use ptx::Ptx;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Symbol<'a> {
    Global(&'a str),
    Device(&'a str),
}

impl<'a> Symbol<'a> {
    pub fn search(code: &'a str) -> impl Iterator<Item = Self> {
        code.split("extern")
            .skip(1)
            .filter_map(|s| s.trim().strip_prefix(r#""C""#))
            .filter_map(|f| f.split_once('(').map(|(head, _)| head.trim()))
            .filter_map(|head| {
                #[inline(always)]
                fn split(head: &str) -> &str {
                    head.rsplit_once(char::is_whitespace).unwrap().1
                }
                if head.contains("__global__") && head.contains("void") {
                    Some(Self::Global(split(head)))
                } else if head.contains("__device__") {
                    Some(Self::Device(split(head)))
                } else {
                    None
                }
            })
    }

    pub fn to_c_string(&self) -> CString {
        match self {
            Self::Global(s) | Self::Device(s) => CString::from_str(s).unwrap(),
        }
    }
}

#[test]
fn test_search_symbols() {
    let code = r#"
extern "C" __global__ void kernel0() { printf("Hello World from GPU!\n"); }
extern "C" __device__ long kernel1() { printf("Hello World from GPU!\n"); }
extern "C" __global__ void kernel2() { printf("Hello World from GPU!\n"); }
    "#;
    assert_eq!(
        Symbol::search(code).collect::<Vec<_>>(),
        &[
            Symbol::Global("kernel0"),
            Symbol::Device("kernel1"),
            Symbol::Global("kernel2"),
        ]
    );
}

#[test]
fn test_behavior() {
    use std::{
        ffi::CString,
        ptr::{null, null_mut},
    };

    let src = r#"extern "C" __global__ void kernel() { printf("Hello World from GPU!\n"); }"#;
    let code = CString::new(src).unwrap();
    let ptx = {
        let mut program = null_mut();
        nvrtc!(nvrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));
        println!("nvrtcCreateProgram passed");
        
        // Create compiler options
        let options = vec![
            CString::new("-I").unwrap(),
            CString::new("/usr/local/corex-4.1.3/lib64/clang/16/include").unwrap(),
            CString::new("-I").unwrap(),
            CString::new("/usr/local/corex-4.1.3/lib64/python3/dist-packages/tensorflow/include/third_party/gpus/cuda/include").unwrap(),
            CString::new("-Xclang").unwrap(),
            CString::new("-fno-cuda-host-device-constexpr").unwrap(),
            CString::new("--no-cuda-version-check").unwrap(),
        ];
        
        if options.len() > 0 {
            let opt_ptrs: Vec<*const u8> = options.iter().map(|opt: &CString| opt.as_ptr()).collect();
            nvrtc!(nvrtcCompileProgram(program, opt_ptrs.len() as i32, opt_ptrs.as_ptr()));
        } else {
            nvrtc!(nvrtcCompileProgram(program, 0, null()));
        }
        println!("nvrtcCompileProgram passed");

        let mut ptx_len = 0;
        nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
        println!("ptx_len = {ptx_len}");

        let mut ptx = vec![0u8; ptx_len];
        nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
        nvrtc!(nvrtcDestroyProgram(&mut program));
        ptx
    };
    let ptx = ptx.as_slice();
    let name = CString::new("kernel").unwrap();

    let mut m = null_mut();
    let mut f = null_mut();

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    crate::Device::new(0).context().apply(|_| {
        driver!(cuModuleLoadData(&mut m, ptx.as_ptr().cast()));
        driver!(cuModuleGetFunction(&mut f, m, name.as_ptr()));
        #[rustfmt::skip]
        driver!(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, null_mut(), null_mut(), null_mut()));
    });
}
