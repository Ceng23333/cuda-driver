fn main() {
    use build_script_cfg::Cfg;
    use find_cuda_helper::{find_cuda_root, include_cuda};
    use search_corex_tools::{find_corex, include_corex};
    use std::{env, path::PathBuf};

    println!("cargo:rerun-if-changed=build.rs");

    let nvidia = Cfg::new("nvidia");
    let iluvatar = Cfg::new("iluvatar");

    enum Vendor {
        Nvidia,
        Iluvatar,
    }

    let (vendor, toolkit) = if let Some(corex) = find_corex() {
        include_corex(&corex);
        iluvatar.define();
        (Vendor::Iluvatar, corex)
    } else if let Some(cuda_root) = find_cuda_root() {
        include_cuda();
        nvidia.define();
        (Vendor::Nvidia, cuda_root)
    } else {
        return;
    };

    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper.h");
    let include = toolkit.join("include");
    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let mut builder = bindgen::Builder::default();
    builder = builder
        // The input header we would like to generate bindings for.
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include.display()))
        // Only generate bindings for the functions in these namespaces.
        .allowlist_function("cublas.*")
        .allowlist_item("cublas.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("cublasStatus_t")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Use core instead of std in the generated bindings.
        .use_core()
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    if let Vendor::Iluvatar = vendor {
        builder = builder
            .clang_args(["-x", "c++"])
            .clang_arg("-I/usr/local/corex/include")
            .clang_arg("-I/usr/local/corex/lib64/clang/16/include")
            .clang_arg("-I/usr/local/corex/include/cuda/std/detail/libcxx/include")
    }
    let bindings = builder
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
