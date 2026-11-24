use serde::{Deserialize, Serialize};

pub const DEFAULT_PDF_DIR: &str = r"C:\Users\2674469\Downloads\テキスト\テキスト";
pub const CACHE_VERSION: u32 = 3;

/// Page structural category classified by the lightweight analysis model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageRole {
    TitleCover,
    Summary,
    Definition,
    ComparisonTable,
    DiagramSlide,
    QuizCheck,
}

impl PageRole {
    pub fn label(&self) -> &'static str {
        match self {
            PageRole::TitleCover => "表紙・扉",
            PageRole::Summary => "概要・まとめ",
            PageRole::Definition => "定義・解説",
            PageRole::ComparisonTable => "比較・一覧",
            PageRole::DiagramSlide => "図解・要点",
            PageRole::QuizCheck => "問題・演習",
        }
    }
}

/// Query intent classified by the lightweight judgment model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    DefinitionLookup,
    ComparisonChoice,
    ConditionConstraint,
    QuestionAnswering,
    GeneralLookup,
}

impl QueryIntent {
    pub fn label(&self) -> &'static str {
        match self {
            QueryIntent::DefinitionLookup => "概念・定義探索",
            QueryIntent::ComparisonChoice => "比較・特徴・使い分け",
            QueryIntent::ConditionConstraint => "条件・制限・構成確認",
            QueryIntent::QuestionAnswering => "問題文・根拠特定",
            QueryIntent::GeneralLookup => "キーワード検索",
        }
    }
}

/// Judgment confidence tier for a search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JudgmentRank {
    S,
    A,
    B,
    C,
}

impl JudgmentRank {
    pub fn label(&self) -> &'static str {
        match self {
            JudgmentRank::S => "S",
            JudgmentRank::A => "A",
            JudgmentRank::B => "B",
            JudgmentRank::C => "C",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDocument {
    pub doc_id: usize,
    pub file_id: u32,
    pub file_name: String,
    pub file_path: String,
    pub file_label: String,
    pub page_num: u32,
    pub page_title: String,
    pub page_role: PageRole,
    pub top_terms: Vec<(String, f32)>,
    pub display_text: String,
    pub search_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfFileInfo {
    pub file_id: u32,
    pub file_label: String,
    pub file_name: String,
    pub file_path: String,
    pub page_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileSignature {
    pub rel_path: String,
    pub size: u64,
    pub modified_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeta {
    pub version: u32,
    pub source_dir: String,
    pub files: Vec<FileSignature>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppSettings {
    pub last_pdf_dir: Option<String>,
    pub recent_dirs: Vec<String>,
    pub recent_queries: Vec<String>,
    pub font_size: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    SmartAnalysis,
    Normal,
    SubstringRegex,
}

/// Extracted keyword with its corpus IDF importance weight.
#[derive(Debug, Clone)]
pub struct WeightedKeyword {
    pub term: String,
    pub idf: f32,
    pub is_compound: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SearchResult {
    pub doc_id: usize,
    pub file_id: u32,
    pub file_label: String,
    pub file_name: String,
    pub file_path: String,
    pub page_num: u32,
    pub page_title: String,
    pub page_role: PageRole,
    pub score: f32,
    pub confidence_pct: u8,
    pub rank: JudgmentRank,
    pub matched_terms: Vec<String>,
    pub missing_terms: Vec<String>,
    pub judgment_reasons: Vec<String>,
    pub best_sentence: String,
    pub snippet: String,
}
