//! Compiles the CUDA core (core/*.cu) with nvcc into a static library and links it.
//! CUDA_HOME selects the toolkit (default /usr/local/cuda).
use std::{env, path::PathBuf, process::Command};

fn main() {
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("engine.o");
    let lib = out.join("libwitness_core.a");

    println!("cargo:rerun-if-changed=core");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // WK_NVCC_FLAGS: extra flags, e.g. "-DWK_GEMV_ROWS=2" to try a kernel tuning constant.
    println!("cargo:rerun-if-env-changed=WK_NVCC_FLAGS");
    let extra = env::var("WK_NVCC_FLAGS").unwrap_or_default();
    run(Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-std=c++17", "-arch=sm_89", "-Xcompiler", "-fPIC"])
        .args(extra.split_whitespace())
        .args(["-c", "core/engine.cu", "-o"])
        .arg(&obj));
    let _ = std::fs::remove_file(&lib);
    run(Command::new("ar").arg("crs").arg(&lib).arg(&obj));

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=witness_core");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed");
}
