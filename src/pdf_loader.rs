use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use pdf_oxide::PdfDocument;
use rayon::prelude::*;
use regex::Regex;

use crate::analyzer_model::{
    analyze_page_structure, detect_repeating_boilerplate, CorpusAnalysisModel,
};
use crate::models::{
    AppSettings, CacheMeta, FileSignature, PageDocument, PdfFileInfo, CACHE_VERSION,
};
use crate::text_transcriber::{build_search_text_from_display, transcribe_page};

/// Recursively scan a directory (including subfolders) for all `.pdf` files.
pub fn scan_pdf_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.exists() {
        return Err(format!("フォルダが見つかりません: {}", dir.display()));
    }
    if dir.is_file() {
        if dir
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| e.eq_ignore_ascii_case("pdf"))
            .unwrap_or(false)
        {
            return Ok(vec![dir.to_path_buf()]);
        }
        return Err(format!("PDFファイルではありません: {}", dir.display()));
    }

    let mut pdfs = Vec::new();
    collect_pdfs_recursive(dir, &mut pdfs, 0)?;
    pdfs.sort();
    Ok(pdfs)
}

fn collect_pdfs_recursive(
    dir: &Path,
    out: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<(), String> {
    if depth > 6 {
        return Ok(());
    }
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("フォルダの読み込みに失敗しました ({}): {}", dir.display(), e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if !name.starts_with('.') && !name.eq_ignore_ascii_case("cache") {
                let _ = collect_pdfs_recursive(&path, out, depth + 1);
            }
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("pdf") {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

/// Build file signatures (relative path, size, mtime) to validate cache.
pub fn compute_file_signatures(base_dir: &Path, pdfs: &[PathBuf]) -> Vec<FileSignature> {
    pdfs.iter()
        .filter_map(|p| {
            let meta = fs::metadata(p).ok()?;
            let rel_path = p
                .strip_prefix(base_dir)
                .unwrap_or(p)
                .to_string_lossy()
                .to_string();
            let modified_secs = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            Some(FileSignature {
                rel_path,
                size: meta.len(),
                modified_secs,
            })
        })
        .collect()
}

/// Generate a friendly display label for any PDF file (AWS Module or general PDF).
pub fn build_file_label(base_dir: &Path, pdf_path: &Path) -> (u32, String) {
    let file_name = pdf_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("document.pdf");

    let re = Regex::new(r"(?i)^AcademyCloudFoundations_Module[_\-\s]*0*(\d+)\.pdf$").unwrap();
    if let Some(caps) = re.captures(file_name) {
        if let Ok(num) = caps[1].parse::<u32>() {
            return (num, format!("Module {:02}", num));
        }
    }

    let stem = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name);

    // If inside a subfolder, prefix with immediate parent folder name
    if let Ok(rel) = pdf_path.strip_prefix(base_dir) {
        if let Some(parent) = rel.parent() {
            if !parent.as_os_str().is_empty() {
                let p_str = parent.to_string_lossy();
                return (9999, format!("{}/{}", p_str, stem));
            }
        }
    }

    // Shorten overly long common prefixes if needed
    let chars: Vec<char> = stem.chars().collect();
    let label = if chars.len() > 28 {
        let head: String = chars[..26].iter().collect();
        format!("{}…", head)
    } else {
        stem.to_string()
    };
    (9999, label)
}

/// Extract all pages from arbitrary PDFs in parallel using Rayon,
/// performing automatic boilerplate detection and structural page classification.
pub fn extract_all_pdfs<F>(
    base_dir: &Path,
    pdfs: &[PathBuf],
    on_progress: F,
) -> Result<(Vec<PageDocument>, Vec<PdfFileInfo>), String>
where
    F: Fn(usize, usize, &str) + Send + Sync,
{
    let total = pdfs.len();
    let completed = Arc::new(AtomicUsize::new(0));

    // Pre-sort PDFs by natural module order or relative path
    let mut ordered_pdfs: Vec<(u32, String, PathBuf)> = pdfs
        .iter()
        .map(|p| {
            let (sort_key, label) = build_file_label(base_dir, p);
            (sort_key, label, p.clone())
        })
        .collect();

    ordered_pdfs.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let per_file_results: Vec<Result<(PdfFileInfo, Vec<PageDocument>), String>> = ordered_pdfs
        .par_iter()
        .enumerate()
        .map(|(file_idx, (_, file_label, path))| {
            let file_id = file_idx as u32;
            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown.pdf")
                .to_string();
            let file_path = path.to_string_lossy().to_string();

            let doc = PdfDocument::open(path)
                .map_err(|e| format!("PDFを開けません ({}): {}", file_name, e))?;
            let page_count = doc
                .page_count()
                .map_err(|e| format!("ページ数取得失敗 ({}): {}", file_name, e))?;

            let mut raw_pages = Vec::with_capacity(page_count);
            for p in 0..page_count {
                raw_pages.push(transcribe_page(&doc, p));
            }

            // Statistical boilerplate detection across this PDF's pages
            let boilerplate = detect_repeating_boilerplate(&raw_pages);

            let mut pages = Vec::with_capacity(page_count);
            for (p, display_text) in raw_pages.into_iter().enumerate() {
                let page_num = (p + 1) as u32;
                let (page_title, page_role) =
                    analyze_page_structure(&display_text, &boilerplate, page_num);
                let search_text = build_search_text_from_display(&display_text, &boilerplate);

                pages.push(PageDocument {
                    doc_id: 0,
                    file_id,
                    file_name: file_name.clone(),
                    file_path: file_path.clone(),
                    file_label: file_label.clone(),
                    page_num,
                    page_title,
                    page_role,
                    top_terms: Vec::new(), // populated by CorpusAnalysisModel
                    display_text,
                    search_text,
                });
            }

            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
            on_progress(done, total, &file_name);

            let info = PdfFileInfo {
                file_id,
                file_label: file_label.clone(),
                file_name,
                file_path,
                page_count: page_count as u32,
            };
            Ok((info, pages))
        })
        .collect();

    let mut all_pages = Vec::new();
    let mut files = Vec::new();
    let mut last_err = None;

    for res in per_file_results {
        match res {
            Ok((info, pages)) => {
                files.push(info);
                all_pages.extend(pages);
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    if files.is_empty() {
        return Err(last_err.unwrap_or_else(|| "有効なPDFを読み込めませんでした".to_string()));
    }

    all_pages.sort_by(|a, b| {
        a.file_id
            .cmp(&b.file_id)
            .then_with(|| a.page_num.cmp(&b.page_num))
    });

    for (i, page) in all_pages.iter_mut().enumerate() {
        page.doc_id = i;
    }

    Ok((all_pages, files))
}

/// Compute a per-folder cache directory using FNV-1a hash of the folder path.
pub fn folder_cache_dir(base_cache_dir: &Path, source_dir: &Path) -> PathBuf {
    let norm = source_dir
        .to_string_lossy()
        .to_lowercase()
        .replace('\\', "/");
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in norm.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    base_cache_dir.join(format!("corpus_{:016x}", hash))
}

pub fn cache_meta_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("meta.json")
}

pub fn cache_pages_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("pages.json")
}

pub fn cache_files_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("files.json")
}

pub fn cache_model_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("analysis_model.json")
}

pub fn cache_index_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join("tantivy_index")
}

pub fn is_cache_valid(
    cache_dir: &Path,
    source_dir: &Path,
    current_sigs: &[FileSignature],
) -> bool {
    let meta_path = cache_meta_path(cache_dir);
    let pages_path = cache_pages_path(cache_dir);
    let files_path = cache_files_path(cache_dir);
    let model_path = cache_model_path(cache_dir);
    let index_dir = cache_index_dir(cache_dir);

    if !meta_path.exists()
        || !pages_path.exists()
        || !files_path.exists()
        || !model_path.exists()
        || !index_dir.exists()
    {
        return false;
    }

    let Ok(meta_str) = fs::read_to_string(&meta_path) else {
        return false;
    };
    let Ok(meta) = serde_json::from_str::<CacheMeta>(&meta_str) else {
        return false;
    };

    meta.version == CACHE_VERSION
        && meta.source_dir == source_dir.to_string_lossy()
        && meta.files == current_sigs
}

pub fn save_corpus_cache(
    cache_dir: &Path,
    source_dir: &Path,
    sigs: &[FileSignature],
    pages: &[PageDocument],
    files: &[PdfFileInfo],
    model: &CorpusAnalysisModel,
) -> Result<(), String> {
    fs::create_dir_all(cache_dir)
        .map_err(|e| format!("キャッシュディレクトリ作成失敗: {}", e))?;

    let meta = CacheMeta {
        version: CACHE_VERSION,
        source_dir: source_dir.to_string_lossy().to_string(),
        files: sigs.to_vec(),
    };

    let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    fs::write(cache_meta_path(cache_dir), meta_json).map_err(|e| e.to_string())?;

    let pages_json = serde_json::to_string(pages).map_err(|e| e.to_string())?;
    fs::write(cache_pages_path(cache_dir), pages_json).map_err(|e| e.to_string())?;

    let files_json = serde_json::to_string_pretty(files).map_err(|e| e.to_string())?;
    fs::write(cache_files_path(cache_dir), files_json).map_err(|e| e.to_string())?;

    let model_json = serde_json::to_string(model).map_err(|e| e.to_string())?;
    fs::write(cache_model_path(cache_dir), model_json).map_err(|e| e.to_string())?;

    Ok(())
}

pub fn load_corpus_cache(
    cache_dir: &Path,
) -> Result<(Vec<PageDocument>, Vec<PdfFileInfo>, CorpusAnalysisModel), String> {
    let pages_str = fs::read_to_string(cache_pages_path(cache_dir))
        .map_err(|e| format!("ページキャッシュ読み込み失敗: {}", e))?;
    let pages: Vec<PageDocument> = serde_json::from_str(&pages_str)
        .map_err(|e| format!("ページキャッシュJSON解析失敗: {}", e))?;

    let files_str = fs::read_to_string(cache_files_path(cache_dir))
        .map_err(|e| format!("ファイルキャッシュ読み込み失敗: {}", e))?;
    let files: Vec<PdfFileInfo> = serde_json::from_str(&files_str)
        .map_err(|e| format!("ファイルキャッシュJSON解析失敗: {}", e))?;

    let model_str = fs::read_to_string(cache_model_path(cache_dir))
        .map_err(|e| format!("解析モデルキャッシュ読み込み失敗: {}", e))?;
    let model: CorpusAnalysisModel = serde_json::from_str(&model_str)
        .map_err(|e| format!("解析モデルキャッシュJSON解析失敗: {}", e))?;

    Ok((pages, files, model))
}

pub fn load_app_settings(base_cache_dir: &Path) -> AppSettings {
    let path = base_cache_dir.join("settings.json");
    if let Ok(s) = fs::read_to_string(path) {
        if let Ok(mut settings) = serde_json::from_str::<AppSettings>(&s) {
            if settings.font_size < 11.0 || settings.font_size > 28.0 {
                settings.font_size = 15.0;
            }
            return settings;
        }
    }
    AppSettings {
        last_pdf_dir: None,
        recent_dirs: Vec::new(),
        recent_queries: Vec::new(),
        font_size: 15.0,
    }
}

pub fn save_app_settings(base_cache_dir: &Path, settings: &AppSettings) {
    let _ = fs::create_dir_all(base_cache_dir);
    let path = base_cache_dir.join("settings.json");
    if let Ok(json) = serde_json::to_string_pretty(settings) {
        let _ = fs::write(path, json);
    }
}
