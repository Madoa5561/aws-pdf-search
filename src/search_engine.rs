use std::collections::HashSet;
use std::fs;
use std::path::Path;

use lindera::analysis::character_filter::unicode_normalize::{
    UnicodeNormalizeCharacterFilter, UnicodeNormalizeKind,
};
use lindera::analysis::character_filter::BoxCharacterFilter;
use lindera::analysis::token_filter::lowercase::LowercaseTokenFilter;
use lindera::analysis::token_filter::BoxTokenFilter;
use lindera::analysis::tokenizer::Tokenizer as LinderaAnalyzer;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera_tantivy::tokenizer::LinderaTokenizer;
use regex::{Regex, RegexBuilder};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{
    IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, INDEXED, STORED,
};
use tantivy::{doc, Index, IndexReader, ReloadPolicy, TantivyDocument, Term};
use unicode_normalization::UnicodeNormalization;

use crate::analyzer_model::{
    classify_query_intent, extract_terms_and_compounds, judge_page_relevance,
    CorpusAnalysisModel, STOPWORDS,
};
use crate::models::{
    JudgmentRank, PageDocument, QueryIntent, SearchResult, WeightedKeyword,
};

const TOKENIZER_NAME: &str = "lang_ja_nfkc_lower";

mod embedded_ipadic {
    lindera_dictionary::embedded_dictionary!("/lindera-ipadic", EmbeddedIPADICLoader);
}

pub struct SearchEngine {
    index: Index,
    reader: IndexReader,
    doc_id_field: tantivy::schema::Field,
    file_id_field: tantivy::schema::Field,
    title_field: tantivy::schema::Field,
    body_field: tantivy::schema::Field,
    question_analyzer: LinderaAnalyzer,
    alnum_re: Regex,
    pub analysis_model: CorpusAnalysisModel,
}

fn build_schema() -> (
    Schema,
    tantivy::schema::Field,
    tantivy::schema::Field,
    tantivy::schema::Field,
    tantivy::schema::Field,
) {
    let mut schema_builder = Schema::builder();
    let doc_id_field = schema_builder.add_u64_field("doc_id", STORED);
    let file_id_field = schema_builder.add_u64_field("file_id", INDEXED | STORED);
    let text_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(TOKENIZER_NAME)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let title_field = schema_builder.add_text_field("title", text_opts.clone());
    let body_field = schema_builder.add_text_field("body", text_opts);
    (
        schema_builder.build(),
        doc_id_field,
        file_id_field,
        title_field,
        body_field,
    )
}

fn create_tantivy_tokenizer() -> Result<LinderaTokenizer, String> {
    let dictionary = embedded_ipadic::load()
        .map_err(|e| format!("IPADIC辞書の読み込みに失敗しました: {:?}", e))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    let mut tokenizer = LinderaTokenizer::from_segmenter(segmenter);
    tokenizer.append_character_filter(BoxCharacterFilter::from(
        UnicodeNormalizeCharacterFilter::new(UnicodeNormalizeKind::NFKC),
    ));
    tokenizer.append_token_filter(BoxTokenFilter::from(LowercaseTokenFilter::new()));
    Ok(tokenizer)
}

fn create_question_analyzer() -> Result<LinderaAnalyzer, String> {
    let dictionary = embedded_ipadic::load()
        .map_err(|e| format!("IPADIC辞書の読み込みに失敗しました: {:?}", e))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    let mut analyzer = LinderaAnalyzer::new(segmenter);
    analyzer.append_character_filter(BoxCharacterFilter::from(
        UnicodeNormalizeCharacterFilter::new(UnicodeNormalizeKind::NFKC),
    ));
    Ok(analyzer)
}

impl SearchEngine {
    /// Train the CorpusAnalysisModel on all pages, annotate pages with top TF-IDF terms,
    /// and build the Tantivy full-text index on disk.
    pub fn build_and_save(
        index_dir: &Path,
        pages: &mut [PageDocument],
    ) -> Result<Self, String> {
        let question_analyzer = create_question_analyzer()?;
        let alnum_re = Regex::new(r"\b[A-Za-z][A-Za-z0-9_\-\.]{1,}\b").unwrap();

        // Train the lightweight statistical analysis model across the entire corpus
        let analysis_model = CorpusAnalysisModel::build_and_annotate_pages(
            pages,
            &question_analyzer,
            &alnum_re,
        );

        if index_dir.exists() {
            fs::remove_dir_all(index_dir)
                .map_err(|e| format!("古いインデックスの削除に失敗: {}", e))?;
        }
        fs::create_dir_all(index_dir)
            .map_err(|e| format!("インデックスディレクトリ作成失敗: {}", e))?;

        let (schema, doc_id_field, file_id_field, title_field, body_field) = build_schema();
        let index = Index::create_in_dir(index_dir, schema)
            .map_err(|e| format!("Tantivyインデックス作成失敗: {}", e))?;

        let tokenizer = create_tantivy_tokenizer()?;
        index.tokenizers().register(TOKENIZER_NAME, tokenizer);

        let mut writer = index
            .writer(50_000_000)
            .map_err(|e| format!("IndexWriter作成失敗: {}", e))?;

        for page in pages.iter() {
            writer
                .add_document(doc!(
                    doc_id_field => page.doc_id as u64,
                    file_id_field => page.file_id as u64,
                    title_field => page.page_title.as_str(),
                    body_field => page.search_text.as_str(),
                ))
                .map_err(|e| format!("ドキュメント追加失敗: {}", e))?;
        }

        writer
            .commit()
            .map_err(|e| format!("インデックスコミット失敗: {}", e))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| format!("IndexReader作成失敗: {}", e))?;

        Ok(Self {
            index,
            reader,
            doc_id_field,
            file_id_field,
            title_field,
            body_field,
            question_analyzer,
            alnum_re,
            analysis_model,
        })
    }

    /// Open an existing Tantivy index + cached CorpusAnalysisModel from disk.
    pub fn open_existing(
        index_dir: &Path,
        analysis_model: CorpusAnalysisModel,
    ) -> Result<Self, String> {
        let index = Index::open_in_dir(index_dir)
            .map_err(|e| format!("キャッシュ済みインデックスを開けません: {}", e))?;

        let tokenizer = create_tantivy_tokenizer()?;
        index.tokenizers().register(TOKENIZER_NAME, tokenizer);

        let schema = index.schema();
        let doc_id_field = schema
            .get_field("doc_id")
            .map_err(|e| format!("doc_idフィールド不明: {}", e))?;
        let file_id_field = schema
            .get_field("file_id")
            .map_err(|e| format!("file_idフィールド不明: {}", e))?;
        let title_field = schema
            .get_field("title")
            .map_err(|e| format!("titleフィールド不明: {}", e))?;
        let body_field = schema
            .get_field("body")
            .map_err(|e| format!("bodyフィールド不明: {}", e))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| format!("IndexReader作成失敗: {}", e))?;

        let question_analyzer = create_question_analyzer()?;
        let alnum_re = Regex::new(r"\b[A-Za-z][A-Za-z0-9_\-\.]{1,}\b").unwrap();

        Ok(Self {
            index,
            reader,
            doc_id_field,
            file_id_field,
            title_field,
            body_field,
            question_analyzer,
            alnum_re,
            analysis_model,
        })
    }

    /// Extract IDF-weighted keywords & compound nouns from any query or exam question.
    pub fn analyze_query_keywords(&self, raw_text: &str) -> Vec<WeightedKeyword> {
        let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();
        let extracted = extract_terms_and_compounds(
            raw_text,
            &self.question_analyzer,
            &self.alnum_re,
            &stopwords,
        );

        let mut seen: HashSet<String> = HashSet::new();
        let mut weighted: Vec<WeightedKeyword> = Vec::new();

        for (surface, lower, is_compound) in extracted {
            if seen.insert(lower.clone()) {
                let mut idf = self.analysis_model.term_idf(&lower);
                if is_compound {
                    idf *= 1.25;
                }
                weighted.push(WeightedKeyword {
                    term: surface,
                    idf,
                    is_compound,
                });
            }
        }

        // If user entered a very short word that was filtered as stopword, fall back to whitespace tokens
        if weighted.is_empty() {
            for tok in raw_text.nfkc().collect::<String>().split_whitespace() {
                let clean = tok.trim_matches(|c: char| c == '"' || c == '-');
                if !clean.is_empty() {
                    let idf = self.analysis_model.term_idf(clean);
                    weighted.push(WeightedKeyword {
                        term: clean.to_string(),
                        idf,
                        is_compound: false,
                    });
                }
            }
        }

        // Sort by IDF descending so the most discriminative terms appear first
        weighted.sort_by(|a, b| b.idf.partial_cmp(&a.idf).unwrap_or(std::cmp::Ordering::Equal));
        weighted
    }

    fn build_file_filter(&self, enabled_files: &HashSet<u32>) -> Box<dyn Query> {
        let clauses: Vec<(Occur, Box<dyn Query>)> = enabled_files
            .iter()
            .map(|&fid| {
                let term = Term::from_field_u64(self.file_id_field, fid as u64);
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>,
                )
            })
            .collect();
        Box::new(BooleanQuery::new(clauses))
    }

    fn parse_single_clause(&self, text: &str, is_phrase: bool) -> Option<Box<dyn Query>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let mut qp =
            QueryParser::for_index(&self.index, vec![self.title_field, self.body_field]);
        qp.set_field_boost(self.title_field, 2.2);
        qp.set_conjunction_by_default();

        let escaped = escape_tantivy_query(trimmed);
        let q_str = if is_phrase {
            format!("\"{}\"", escaped)
        } else {
            escaped
        };
        qp.parse_query(&q_str).ok()
    }

    /// Smart Analysis Mode: combines morphological + compound noun extraction,
    /// Corpus IDF weighting, and the 6-feature Relevance Judgment Model.
    pub fn search_smart_analysis(
        &self,
        raw_input: &str,
        active_keywords: &[WeightedKeyword],
        enabled_files: &HashSet<u32>,
        pages: &[PageDocument],
        limit: usize,
    ) -> (Vec<SearchResult>, Vec<String>, QueryIntent) {
        let intent = classify_query_intent(raw_input);
        if active_keywords.is_empty() || enabled_files.is_empty() {
            return (Vec::new(), Vec::new(), intent);
        }

        let mut kw_clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        let mut highlight_terms: Vec<String> = Vec::new();

        for kw in active_keywords {
            if let Some(q) = self.parse_single_clause(&kw.term, kw.is_compound) {
                kw_clauses.push((Occur::Should, q));
                highlight_terms.push(kw.term.clone());
            } else if let Some(q) = self.parse_single_clause(&kw.term, false) {
                kw_clauses.push((Occur::Should, q));
                highlight_terms.push(kw.term.clone());
            }
        }

        if kw_clauses.is_empty() {
            return (Vec::new(), Vec::new(), intent);
        }

        let root_clauses = vec![
            (
                Occur::Must,
                Box::new(BooleanQuery::new(kw_clauses)) as Box<dyn Query>,
            ),
            (Occur::Must, self.build_file_filter(enabled_files)),
        ];
        let final_query = BooleanQuery::new(root_clauses);
        let results = self.execute_judged_tantivy_query(
            &final_query,
            pages,
            active_keywords,
            &highlight_terms,
            intent,
            limit,
        );
        (results, highlight_terms, intent)
    }

    /// Normal keyword search (AND/OR, "phrase", -exclude), enriched with judgment model scoring.
    pub fn search_normal(
        &self,
        raw_query: &str,
        use_or: bool,
        enabled_files: &HashSet<u32>,
        pages: &[PageDocument],
        limit: usize,
    ) -> (Vec<SearchResult>, Vec<String>) {
        if raw_query.trim().is_empty() || enabled_files.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let parsed = parse_normal_query_tokens(raw_query);
        let mut root_clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        let mut positive_clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        let mut highlight_terms: Vec<String> = Vec::new();
        let mut weighted_kws: Vec<WeightedKeyword> = Vec::new();

        for phrase in &parsed.phrases {
            if let Some(q) = self.parse_single_clause(phrase, true) {
                positive_clauses.push((if use_or { Occur::Should } else { Occur::Must }, q));
                highlight_terms.push(phrase.clone());
                weighted_kws.push(WeightedKeyword {
                    term: phrase.clone(),
                    idf: self.analysis_model.term_idf(phrase) * 1.3,
                    is_compound: true,
                });
            }
        }
        for term in &parsed.terms {
            if let Some(q) = self.parse_single_clause(term, false) {
                positive_clauses.push((if use_or { Occur::Should } else { Occur::Must }, q));
                highlight_terms.push(term.clone());
                weighted_kws.push(WeightedKeyword {
                    term: term.clone(),
                    idf: self.analysis_model.term_idf(term),
                    is_compound: false,
                });
            }
        }

        if positive_clauses.is_empty() {
            return (Vec::new(), Vec::new());
        }

        if use_or {
            root_clauses.push((Occur::Must, Box::new(BooleanQuery::new(positive_clauses))));
        } else {
            root_clauses.extend(positive_clauses);
        }

        for ex in &parsed.excludes {
            let is_phrase = ex.contains(' ');
            if let Some(q) = self.parse_single_clause(ex, is_phrase) {
                root_clauses.push((Occur::MustNot, q));
            }
        }

        root_clauses.push((Occur::Must, self.build_file_filter(enabled_files)));

        let intent = classify_query_intent(raw_query);
        let final_query = BooleanQuery::new(root_clauses);
        let results = self.execute_judged_tantivy_query(
            &final_query,
            pages,
            &weighted_kws,
            &highlight_terms,
            intent,
            limit,
        );
        (results, highlight_terms)
    }

    fn execute_judged_tantivy_query(
        &self,
        query: &dyn Query,
        pages: &[PageDocument],
        query_keywords: &[WeightedKeyword],
        highlight_terms: &[String],
        intent: QueryIntent,
        limit: usize,
    ) -> Vec<SearchResult> {
        let searcher = self.reader.searcher();
        let candidate_limit = (limit * 2).max(100);
        let Ok(top_docs) =
            searcher.search(query, &TopDocs::with_limit(candidate_limit).order_by_score())
        else {
            return Vec::new();
        };

        let max_bm25 = top_docs.first().map(|(s, _)| *s).unwrap_or(1.0);
        let mut results = Vec::with_capacity(top_docs.len());

        for (bm25_score, doc_addr) in top_docs {
            let Ok(retrieved) = searcher.doc::<TantivyDocument>(doc_addr) else {
                continue;
            };
            let Some(doc_id) = retrieved
                .get_first(self.doc_id_field)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
            else {
                continue;
            };
            if let Some(page) = pages.get(doc_id) {
                let judgment =
                    judge_page_relevance(page, bm25_score, max_bm25, query_keywords, intent);
                let snippet = if !judgment.best_sentence.is_empty() {
                    judgment.best_sentence.clone()
                } else {
                    build_snippet(&page.display_text, highlight_terms, None)
                };

                results.push(SearchResult {
                    doc_id: page.doc_id,
                    file_id: page.file_id,
                    file_label: page.file_label.clone(),
                    file_name: page.file_name.clone(),
                    file_path: page.file_path.clone(),
                    page_num: page.page_num,
                    page_title: page.page_title.clone(),
                    page_role: page.page_role,
                    score: judgment.final_score,
                    confidence_pct: judgment.confidence_pct,
                    rank: judgment.rank,
                    matched_terms: judgment.matched_terms,
                    missing_terms: judgment.missing_terms,
                    judgment_reasons: judgment.reasons,
                    best_sentence: judgment.best_sentence,
                    snippet,
                });
            }
        }

        // Re-rank by composite judgment score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.confidence_pct.cmp(&a.confidence_pct))
                .then_with(|| a.file_id.cmp(&b.file_id))
                .then_with(|| a.page_num.cmp(&b.page_num))
        });
        results.truncate(limit);
        results
    }

    /// Direct substring or regex scan across cached pages.
    pub fn search_substring_regex(
        &self,
        raw_query: &str,
        use_regex: bool,
        enabled_files: &HashSet<u32>,
        pages: &[PageDocument],
        limit: usize,
    ) -> Result<(Vec<SearchResult>, Vec<String>), String> {
        let trimmed = raw_query.trim();
        if trimmed.is_empty() || enabled_files.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let nfkc_query: String = trimmed.nfkc().collect();
        let mut results = Vec::new();

        if use_regex {
            let re = RegexBuilder::new(&nfkc_query)
                .case_insensitive(true)
                .build()
                .map_err(|e| format!("正規表現エラー: {}", e))?;

            for page in pages {
                if !enabled_files.contains(&page.file_id) {
                    continue;
                }
                let count = re.find_iter(&page.search_text).count();
                if count > 0 {
                    let snippet = build_snippet(&page.display_text, &[], Some(&re));
                    let rank = if count >= 3 {
                        JudgmentRank::S
                    } else if count == 2 {
                        JudgmentRank::A
                    } else {
                        JudgmentRank::B
                    };
                    results.push(SearchResult {
                        doc_id: page.doc_id,
                        file_id: page.file_id,
                        file_label: page.file_label.clone(),
                        file_name: page.file_name.clone(),
                        file_path: page.file_path.clone(),
                        page_num: page.page_num,
                        page_title: page.page_title.clone(),
                        page_role: page.page_role,
                        score: count as f32,
                        confidence_pct: (65 + (count * 10).min(34)) as u8,
                        rank,
                        matched_terms: vec![nfkc_query.clone()],
                        missing_terms: Vec::new(),
                        judgment_reasons: vec![
                            format!("正規表現一致 {}箇所", count),
                            format!("種別:{}", page.page_role.label()),
                        ],
                        best_sentence: snippet.clone(),
                        snippet,
                    });
                }
            }
        } else {
            let needle = nfkc_query.to_lowercase();
            let highlight_terms = vec![nfkc_query.clone()];
            for page in pages {
                if !enabled_files.contains(&page.file_id) {
                    continue;
                }
                let hay = page.search_text.to_lowercase();
                let count = hay.matches(&needle).count();
                if count > 0 {
                    let title_hit = page
                        .page_title
                        .nfkc()
                        .collect::<String>()
                        .to_lowercase()
                        .contains(&needle);
                    let snippet = build_snippet(&page.display_text, &highlight_terms, None);
                    let rank = if title_hit || count >= 3 {
                        JudgmentRank::S
                    } else if count == 2 {
                        JudgmentRank::A
                    } else {
                        JudgmentRank::B
                    };
                    let mut reasons = vec![format!("部分一致 {}箇所", count)];
                    if title_hit {
                        reasons.push("見出し一致".to_string());
                    }
                    reasons.push(format!("種別:{}", page.page_role.label()));

                    results.push(SearchResult {
                        doc_id: page.doc_id,
                        file_id: page.file_id,
                        file_label: page.file_label.clone(),
                        file_name: page.file_name.clone(),
                        file_path: page.file_path.clone(),
                        page_num: page.page_num,
                        page_title: page.page_title.clone(),
                        page_role: page.page_role,
                        score: count as f32 + if title_hit { 1.5 } else { 0.0 },
                        confidence_pct: (70 + (count * 8).min(28)) as u8,
                        rank,
                        matched_terms: highlight_terms.clone(),
                        missing_terms: Vec::new(),
                        judgment_reasons: reasons,
                        best_sentence: snippet.clone(),
                        snippet,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.file_id.cmp(&b.file_id))
                .then_with(|| a.page_num.cmp(&b.page_num))
        });
        results.truncate(limit);
        Ok((results, vec![nfkc_query]))
    }
}

struct ParsedNormalQuery {
    terms: Vec<String>,
    phrases: Vec<String>,
    excludes: Vec<String>,
}

fn parse_normal_query_tokens(input: &str) -> ParsedNormalQuery {
    let nfkc: String = input.nfkc().collect();
    let chars: Vec<char> = nfkc.chars().collect();
    let mut terms = Vec::new();
    let mut phrases = Vec::new();
    let mut excludes = Vec::new();

    let mut i = 0;
    while i < chars.len() {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }

        let is_exclude = chars[i] == '-' && i + 1 < chars.len() && !chars[i + 1].is_whitespace();
        if is_exclude {
            i += 1;
        }

        if i < chars.len() && chars[i] == '"' {
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            let phrase: String = chars[start..i].iter().collect();
            if i < chars.len() && chars[i] == '"' {
                i += 1;
            }
            let trimmed = phrase.trim().to_string();
            if !trimmed.is_empty() {
                if is_exclude {
                    excludes.push(trimmed);
                } else {
                    phrases.push(trimmed);
                }
            }
        } else {
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            let tok: String = chars[start..i].iter().collect();
            if !tok.is_empty() {
                if is_exclude {
                    excludes.push(tok);
                } else {
                    terms.push(tok);
                }
            }
        }
    }

    ParsedNormalQuery {
        terms,
        phrases,
        excludes,
    }
}

fn escape_tantivy_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if matches!(
            c,
            '+' | '^'
                | '`'
                | ':'
                | '{'
                | '}'
                | '"'
                | '['
                | ']'
                | '('
                | ')'
                | '~'
                | '!'
                | '\\'
                | '*'
                | '-'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Build a flexible regex for highlighting terms in display text,
/// allowing optional linebreaks inside words (e.g. "データベー\nス").
pub fn build_highlight_regex(terms: &[String], raw_regex: Option<&str>) -> Option<Regex> {
    if let Some(pat) = raw_regex {
        return RegexBuilder::new(pat).case_insensitive(true).build().ok();
    }
    let mut patterns: Vec<String> = Vec::new();
    for t in terms {
        let trimmed = t.trim();
        if trimmed.is_empty() {
            continue;
        }
        let chars: Vec<char> = trimmed.chars().collect();
        if chars.len() <= 24 {
            let mut pat = String::new();
            for (idx, ch) in chars.iter().enumerate() {
                if idx > 0 {
                    pat.push_str(r"\n?");
                }
                pat.push_str(&regex::escape(&ch.to_string()));
            }
            patterns.push(pat);
        } else {
            patterns.push(regex::escape(trimmed));
        }
    }
    if patterns.is_empty() {
        return None;
    }
    patterns.sort_by(|a, b| b.len().cmp(&a.len()));
    let combined = patterns.join("|");
    RegexBuilder::new(&combined)
        .case_insensitive(true)
        .build()
        .ok()
}

/// Generate a compact snippet around the earliest match in display_text.
pub fn build_snippet(display_text: &str, terms: &[String], custom_re: Option<&Regex>) -> String {
    let cleaned = display_text
        .lines()
        .filter(|l| !l.contains("Amazon Web Services, Inc. or its affiliates"))
        .collect::<Vec<_>>()
        .join(" ");

    let owned_re;
    let re_opt = if let Some(r) = custom_re {
        Some(r)
    } else {
        owned_re = build_highlight_regex(terms, None);
        owned_re.as_ref()
    };

    let match_byte = re_opt
        .and_then(|re| re.find(&cleaned))
        .map(|m| m.start())
        .unwrap_or(0);

    let match_char_idx = cleaned[..match_byte].chars().count();
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let window_before = 35;
    let window_after = 95;
    let start = match_char_idx.saturating_sub(window_before);
    let end = (match_char_idx + window_after).min(chars.len());

    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("...");
    }
    snippet.extend(&chars[start..end]);
    if end < chars.len() {
        snippet.push_str("...");
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DEFAULT_PDF_DIR;
    use crate::pdf_loader::{extract_all_pdfs, scan_pdf_files};
    use std::path::PathBuf;

    #[test]
    fn test_real_pdfs_analysis_and_judgment_model() {
        let dir = PathBuf::from(DEFAULT_PDF_DIR);
        if !dir.exists() {
            return;
        }
        let pdfs = scan_pdf_files(&dir).expect("scan pdfs");
        assert_eq!(pdfs.len(), 10, "expected 10 module PDFs");

        let (mut pages, files) =
            extract_all_pdfs(&dir, &pdfs, |_, _, _| {}).expect("extract all pdfs");
        assert_eq!(files.len(), 10);
        assert!(pages.len() > 200);

        let tmp_idx = std::env::temp_dir().join("aws_pdf_search_test_idx_v2");
        let engine = SearchEngine::build_and_save(&tmp_idx, &mut pages).expect("build index");
        let all_files: HashSet<u32> = files.iter().map(|f| f.file_id).collect();

        // Verify page top terms & structural roles were populated
        assert!(pages.iter().any(|p| !p.top_terms.is_empty()));
        assert!(pages.iter().any(|p| !p.page_title.is_empty()));

        // 1. Smart Analysis search with full exam question
        let q = "次のうち、AWSの責任共有モデルにおいて顧客の責任範囲に含まれるものはどれか。選択してください。";
        let kws = engine.analyze_query_keywords(q);
        assert!(
            kws.iter().any(|k| k.term.contains("責任共有モデル") || k.term.contains("責任")),
            "extracted keywords: {:?}",
            kws
        );
        let (res_smart, _, intent) =
            engine.search_smart_analysis(q, &kws, &all_files, &pages, 15);
        assert_eq!(intent, QueryIntent::QuestionAnswering);
        assert!(!res_smart.is_empty());
        assert!(res_smart[0].confidence_pct >= 60);

        // 2. Co-occurrence recommendation
        let related = engine
            .analysis_model
            .recommend_related_terms(&["VPC".to_string()], 5);
        assert!(!related.is_empty(), "expected related terms for VPC");

        // 3. Normal & Substring/Regex search
        let (res_vpc, _) = engine.search_normal("ＶＰＣ サブネット", false, &all_files, &pages, 20);
        assert!(!res_vpc.is_empty());

        let (res_re, _) = engine
            .search_substring_regex(r"EC2|S3", true, &all_files, &pages, 20)
            .expect("regex");
        assert!(!res_re.is_empty());

        let _ = fs::remove_dir_all(&tmp_idx);
    }
}
