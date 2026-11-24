use std::fs;
use std::path::PathBuf;
use std::process::Command;

use lindera_dictionary::builder::DictionaryBuilder;
use lindera_dictionary::dictionary::metadata::Metadata;

const IPADIC_URL: &str = "https://Lindera.dev/mecab-ipadic-2.7.0-20250920.tar.gz";

fn main() {
    println!("cargo:rerun-if-changed=ipadic_data/metadata.json");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let output_dir = out_dir.join("lindera-ipadic");

    println!("cargo:rustc-env=LINDERA_WORK_DIR={}", out_dir.display());

    if output_dir.join("dict.trie").exists() && output_dir.join("metadata.json").exists() {
        return;
    }

    let ipadic_data_dir = manifest_dir.join("ipadic_data");
    let metadata_path = ipadic_data_dir.join("metadata.json");
    let input_dir = ipadic_data_dir.join("mecab-ipadic-2.7.0-20250920");

    if !input_dir.exists() {
        fs::create_dir_all(&ipadic_data_dir).expect("Failed to create ipadic_data dir");
        let archive_path = ipadic_data_dir.join("ipadic.tar.gz");
        if !archive_path.exists() {
            let status = Command::new("curl")
                .args([
                    "-fSL",
                    IPADIC_URL,
                    "-o",
                    &archive_path.to_string_lossy(),
                ])
                .status()
                .expect("Failed to execute curl to download IPADIC dictionary");
            assert!(status.success(), "Failed to download IPADIC archive");
        }
        let status = Command::new("tar")
            .args([
                "-xzf",
                &archive_path.to_string_lossy(),
                "-C",
                &ipadic_data_dir.to_string_lossy(),
            ])
            .status()
            .expect("Failed to execute tar to extract IPADIC dictionary");
        assert!(status.success(), "Failed to extract IPADIC archive");
    }

    let metadata_bytes =
        fs::read(&metadata_path).expect("Failed to read ipadic_data/metadata.json");
    let metadata =
        Metadata::load(&metadata_bytes).expect("Failed to parse ipadic_data/metadata.json");

    fs::create_dir_all(&output_dir).expect("Failed to create lindera-ipadic output dir");

    let builder = DictionaryBuilder::new(metadata);
    builder
        .build_dictionary(&input_dir, &output_dir)
        .expect("Failed to build IPADIC dictionary");
}
