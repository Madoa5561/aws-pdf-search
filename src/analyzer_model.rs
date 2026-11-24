use std::collections::{HashMap, HashSet};

use lindera::analysis::tokenizer::Tokenizer as LinderaAnalyzer;
use regex::Regex;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::models::{
    JudgmentRank, PageDocument, PageRole, QueryIntent, WeightedKeyword,
};

/// Stopwords filtered out during automatic keyword & concept extraction.
pub const STOPWORDS: &[&str] = &[
    "次", "うち", "どれ", "どれか", "選択", "ください", "下さい", "正しい", "適切", "最も",
    "以下", "記述", "説明", "該当", "すべて", "全て", "選ん", "選ぶ", "選び", "組み合わせ",
    "問題", "質問", "答え", "解答", "回答", "当てはまる", "当てはまり", "項目", "文",
    "ステートメント", "中", "こと", "もの", "ため", "よう", "場合", "これ", "それ", "あれ",
    "どの", "何", "なに", "方法", "特徴", "内容", "対象", "利用", "使用", "可能", "必要",
    "提供", "実行", "作成", "確認", "つ", "個", "点", "際", "上", "下", "内", "外", "間",
    "他", "等", "など", "について", "において", "による", "に対して", "として", "および",
    "また", "または", "しかし", "ただし", "なお", "例", "とき", "時", "さん", "様",
    "ページ", "スライド", "モジュール", "章", "節", "図", "表", "一覧", "概要", "入門",
    "目標", "まとめ", "知識", "テスト", "ラボ", "アクティビティ", "セクション",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "or", "and", "not", "that", "this", "which",
    "what", "following", "select", "choose", "true", "false", "best", "most", "all",
    "inc", "affiliates", "rights", "reserved", "copyright", "module", "page",
];

/// Lightweight statistical corpus analysis model saved in cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusAnalysisModel {
    pub total_pages: usize,
    /// Normalized lowercase term -> IDF weight
    pub idf_map: HashMap<String, f32>,
    /// Normalized lowercase term -> Canonical display form (e.g. "ec2" -> "EC2")
    pub canonical_form: HashMap<String, String>,
    /// Normalized lowercase term -> Top co-occurring terms (term_lower, PMI/Jaccard score)
    pub cooccurrence: HashMap<String, Vec<(String, f32)>>,
    pub default_idf: f32,
}

/// Detect repeating boilerplate header/footer lines within a single PDF's raw pages.
/// Any normalized line appearing on >= 35% of pages (and at least 3 pages) is marked as noise.
pub fn detect_repeating_boilerplate(raw_pages: &[String]) -> HashSet<String> {
    let mut line_page_counts: HashMap<String, usize> = HashMap::new();
    let total_pages = raw_pages.len();
    if total_pages < 4 {
        return HashSet::new();
    }

    for raw in raw_pages {
        let mut seen_in_page = HashSet::new();
        for line in raw.lines() {
            let trimmed = normalize_boilerplate_key(line);
            if trimmed.len() >= 6 {
                seen_in_page.insert(trimmed);
            }
        }
        for key in seen_in_page {
            *line_page_counts.entry(key).or_insert(0) += 1;
        }
    }

    let threshold = ((total_pages as f32 * 0.35).ceil() as usize).max(3);
    line_page_counts
        .into_iter()
        .filter_map(|(k, count)| if count >= threshold { Some(k) } else { None })
        .collect()
}

pub fn normalize_boilerplate_key(line: &str) -> String {
    // Strip trailing page digits so footer lines with page numbers still match
    let trimmed = line.trim().trim_end_matches(|c: char| c.is_ascii_digit() || c.is_whitespace());
    trimmed.nfkc().collect::<String>().to_lowercase()
}

/// Structural page parser: automatically infers the page's heading/title and structural role.
pub fn analyze_page_structure(
    display_text: &str,
    boilerplate: &HashSet<String>,
    page_num: u32,
) -> (String, PageRole) {
    let meaningful_lines: Vec<&str> = display_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            if l.is_empty() {
                return false;
            }
            let key = normalize_boilerplate_key(l);
            !boilerplate.contains(&key)
                && !l.contains("All rights reserved")
                && !l.chars().all(|c| c.is_ascii_digit() || c.is_whitespace())
        })
        .collect();

    // 1. Infer Page Title from the first prominent non-bullet line
    let mut page_title = String::new();
    for line in meaningful_lines.iter().take(4) {
        let cleaned = line
            .trim_start_matches(|c: char| matches!(c, '•' | '・' | '-' | '*' | '#'))
            .trim();
        let char_len = cleaned.chars().count();
        if (2..=65).contains(&char_len) {
            page_title = cleaned.to_string();
            break;
        }
    }
    if page_title.is_empty() {
        page_title = meaningful_lines
            .first()
            .map(|l| l.chars().take(50).collect())
            .unwrap_or_else(|| format!("Page {}", page_num));
    }

    // 2. Classify PageRole from structural & lexical features
    let total_chars: usize = meaningful_lines.iter().map(|l| l.chars().count()).sum();
    let line_count = meaningful_lines.len();
    let bullet_lines = meaningful_lines
        .iter()
        .filter(|l| l.starts_with('•') || l.starts_with('・') || l.starts_with("- "))
        .count();
    let joined = meaningful_lines.join("\n");
    let lower = joined.to_lowercase();

    let is_quiz = lower.contains("知識確認")
        || lower.contains("問題") && (lower.contains("選択") || lower.contains("どれですか") || lower.contains("どれか"))
        || lower.contains("クイズ")
        || lower.contains("演習")
        || lower.contains("question")
        || joined.matches('？').count() + joined.matches('?').count() >= 2;

    let is_summary = page_title.contains("概要")
        || page_title.contains("まとめ")
        || page_title.contains("目標")
        || page_title.contains("目次")
        || page_title.to_lowercase().contains("summary")
        || page_title.to_lowercase().contains("overview")
        || page_title.contains("重要なポイント");

    let is_definition = page_title.contains("とは")
        || page_title.contains("定義")
        || joined.contains("とは、")
        || joined.contains("は、") && (joined.contains("サービスです") || joined.contains("機能です") || joined.contains("モデルです") || joined.contains("仕組みです"));

    let short_lines = meaningful_lines
        .iter()
        .filter(|l| l.chars().count() <= 18 && !l.starts_with('•'))
        .count();

    let role = if (page_num == 1 || total_chars < 75) && line_count <= 5 {
        PageRole::TitleCover
    } else if is_quiz {
        PageRole::QuizCheck
    } else if is_summary {
        PageRole::Summary
    } else if is_definition {
        PageRole::Definition
    } else if line_count >= 10 && short_lines * 2 >= line_count {
        PageRole::ComparisonTable
    } else if total_chars < 170 && bullet_lines <= 2 {
        PageRole::DiagramSlide
    } else {
        PageRole::Definition
    };

    (page_title, role)
}

/// Extract domain terms & compound nouns from text using Lindera + alphanumeric regex.
/// Returns a list of `(canonical_surface, normalized_lower, is_compound)` in appearance order.
pub fn extract_terms_and_compounds(
    text: &str,
    analyzer: &LinderaAnalyzer,
    alnum_re: &Regex,
    stopwords: &HashSet<&str>,
) -> Vec<(String, String, bool)> {
    let nfkc: String = text.nfkc().collect();
    let mut results: Vec<(String, String, bool)> = Vec::new();

    // 1. Intact alphanumeric terms (e.g. EC2, S3, Route53, CloudFront, IPv4, t2.micro)
    let mut alnum_set: HashSet<String> = HashSet::new();
    for m in alnum_re.find_iter(&nfkc) {
        let word = m
            .as_str()
            .trim_matches(|c: char| matches!(c, '.' | '-' | '_'));
        if word.len() < 2 || word.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let lower = word.to_lowercase();
        if stopwords.contains(lower.as_str()) {
            continue;
        }
        alnum_set.insert(lower.clone());
        results.push((word.to_string(), lower, false));
    }

    // 2. Lindera morphological analysis + Compound Noun (複合名詞) concatenation
    if let Ok(tokens) = analyzer.tokenize(&nfkc) {
        let mut noun_run: Vec<String> = Vec::new();

        let flush_run = |run: &mut Vec<String>,
                         out: &mut Vec<(String, String, bool)>,
                         stopwords: &HashSet<&str>,
                         alnum_set: &HashSet<String>| {
            if run.is_empty() {
                return;
            }
            // If 2..=4 consecutive nouns, also emit the concatenated compound noun!
            if (2..=4).contains(&run.len()) {
                let compound: String = run.concat();
                let char_cnt = compound.chars().count();
                if (3..=22).contains(&char_cnt) {
                    let lower = compound.to_lowercase();
                    if !stopwords.contains(lower.as_str()) {
                        out.push((compound, lower, true));
                    }
                }
            }
            // Also emit individual valid nouns
            for single in run.drain(..) {
                let lower = single.to_lowercase();
                if stopwords.contains(lower.as_str()) {
                    continue;
                }
                if single.is_ascii()
                    && alnum_set
                        .iter()
                        .any(|a| a.len() > lower.len() && a.contains(&lower))
                {
                    continue;
                }
                let char_cnt = single.chars().count();
                if char_cnt == 1 {
                    let c = single.chars().next().unwrap();
                    if ('\u{3040}'..='\u{309F}').contains(&c)
                        || c.is_ascii_digit()
                        || !c.is_alphanumeric()
                    {
                        continue;
                    }
                }
                out.push((single, lower, false));
            }
        };

        for mut token in tokens {
            let surface = token.surface.trim().to_string();
            let pos = token.get_detail(0).unwrap_or("").to_string();
            let subpos = token.get_detail(1).unwrap_or("").to_string();

            let is_content_noun = pos == "名詞"
                && !matches!(
                    subpos.as_str(),
                    "非自立" | "代名詞" | "数" | "特殊" | "副詞可能" | "接続詞的"
                );
            let is_alnum_unk = pos == "UNK"
                && surface.chars().any(|c| c.is_alphanumeric())
                && !surface.chars().all(|c| c.is_ascii_digit() || c.is_ascii_punctuation());

            if (is_content_noun || is_alnum_unk)
                && !surface.is_empty()
                && !surface.chars().all(|c| !c.is_alphanumeric())
            {
                noun_run.push(surface);
            } else {
                flush_run(&mut noun_run, &mut results, stopwords, &alnum_set);
            }
        }
        flush_run(&mut noun_run, &mut results, stopwords, &alnum_set);
    }

    results
}

impl CorpusAnalysisModel {
    /// Train the statistical TF-IDF, canonical term dictionary, page top-terms,
    /// and co-occurrence recommendation graph across all extracted pages.
    pub fn build_and_annotate_pages(
        pages: &mut [PageDocument],
        analyzer: &LinderaAnalyzer,
        alnum_re: &Regex,
    ) -> Self {
        let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();
        let total_pages = pages.len().max(1);

        let mut df_map: HashMap<String, usize> = HashMap::new();
        let mut canonical_form: HashMap<String, String> = HashMap::new();
        let mut per_page_tf: Vec<HashMap<String, f32>> = Vec::with_capacity(pages.len());

        for page in pages.iter() {
            let extracted =
                extract_terms_and_compounds(&page.search_text, analyzer, alnum_re, &stopwords);
            let title_lower = page.page_title.nfkc().collect::<String>().to_lowercase();

            let mut tf: HashMap<String, f32> = HashMap::new();
            for (surface, lower, is_compound) in extracted {
                canonical_form.entry(lower.clone()).or_insert(surface);
                let mut weight = if is_compound { 1.35 } else { 1.0 };
                if title_lower.contains(&lower) {
                    weight += 0.75;
                }
                *tf.entry(lower).or_insert(0.0) += weight;
            }

            for term in tf.keys() {
                *df_map.entry(term.clone()).or_insert(0) += 1;
            }
            per_page_tf.push(tf);
        }

        // Compute smoothed BM25-style IDF for every term in the corpus
        let n = total_pages as f32;
        let mut idf_map: HashMap<String, f32> = HashMap::with_capacity(df_map.len());
        for (term, df) in &df_map {
            let df_f = *df as f32;
            let idf = (1.0 + (n - df_f + 0.5) / (df_f + 0.5)).ln().max(0.15);
            idf_map.insert(term.clone(), idf);
        }
        let default_idf = (1.0 + n).ln().max(1.5);

        // Annotate each page with its top 6 TF-IDF keywords & build co-occurrence counts
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

        for (page, tf) in pages.iter_mut().zip(per_page_tf.into_iter()) {
            let mut scored: Vec<(String, f32)> = tf
                .into_iter()
                .filter_map(|(term, freq)| {
                    let df = df_map.get(&term).copied().unwrap_or(1);
                    // Ignore ultra-common boilerplate terms that appear on >60% of all pages
                    if total_pages >= 10 && df * 10 > total_pages * 6 {
                        return None;
                    }
                    let idf = idf_map.get(&term).copied().unwrap_or(1.0);
                    let score = (1.0 + freq.ln()) * idf;
                    Some((term, score))
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top 8 terms on this page for co-occurrence graph
            let top_for_cooc: Vec<String> =
                scored.iter().take(8).map(|(t, _)| t.clone()).collect();
            for i in 0..top_for_cooc.len() {
                for j in (i + 1)..top_for_cooc.len() {
                    let (a, b) = if top_for_cooc[i] <= top_for_cooc[j] {
                        (top_for_cooc[i].clone(), top_for_cooc[j].clone())
                    } else {
                        (top_for_cooc[j].clone(), top_for_cooc[i].clone())
                    };
                    *pair_counts.entry((a, b)).or_insert(0) += 1;
                }
            }

            page.top_terms = scored
                .into_iter()
                .take(6)
                .map(|(t, s)| {
                    let disp = canonical_form.get(&t).cloned().unwrap_or(t);
                    (disp, s)
                })
                .collect();
        }

        // Build top co-occurring terms map (for related keyword recommendations)
        let mut cooccurrence: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for ((a, b), co_cnt) in pair_counts {
            if co_cnt < 2 && total_pages > 15 {
                continue;
            }
            let df_a = df_map.get(&a).copied().unwrap_or(1) as f32;
            let df_b = df_map.get(&b).copied().unwrap_or(1) as f32;
            let jaccard = (co_cnt as f32) / (df_a + df_b - co_cnt as f32).max(1.0);
            let idf_a = idf_map.get(&a).copied().unwrap_or(1.0);
            let idf_b = idf_map.get(&b).copied().unwrap_or(1.0);

            cooccurrence
                .entry(a.clone())
                .or_default()
                .push((b.clone(), jaccard * idf_b));
            cooccurrence
                .entry(b)
                .or_default()
                .push((a, jaccard * idf_a));
        }

        for list in cooccurrence.values_mut() {
            list.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(6);
        }

        Self {
            total_pages,
            idf_map,
            canonical_form,
            cooccurrence,
            default_idf,
        }
    }

    pub fn term_idf(&self, term: &str) -> f32 {
        let lower = term.nfkc().collect::<String>().to_lowercase();
        self.idf_map
            .get(&lower)
            .copied()
            .unwrap_or(self.default_idf * 0.7)
    }

    /// Recommend related co-occurring domain keywords for the current active keywords.
    pub fn recommend_related_terms(&self, active_terms: &[String], limit: usize) -> Vec<String> {
        let active_lower: HashSet<String> = active_terms
            .iter()
            .map(|t| t.nfkc().collect::<String>().to_lowercase())
            .collect();

        let mut candidate_scores: HashMap<String, f32> = HashMap::new();
        for lower in &active_lower {
            if let Some(neighbors) = self.cooccurrence.get(lower) {
                for (nb, score) in neighbors {
                    if !active_lower.contains(nb)
                        && !active_lower.iter().any(|a| a.contains(nb) || nb.contains(a))
                    {
                        *candidate_scores.entry(nb.clone()).or_insert(0.0) += *score;
                    }
                }
            }
        }

        let mut ranked: Vec<(String, f32)> = candidate_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(limit)
            .map(|(lower, _)| {
                self.canonical_form
                    .get(&lower)
                    .cloned()
                    .unwrap_or(lower)
            })
            .collect()
    }
}

/// Classify user query intent from lexical cues and length.
pub fn classify_query_intent(raw_query: &str) -> QueryIntent {
    let q: String = raw_query.nfkc().collect();
    let lower = q.to_lowercase();
    let char_cnt = q.chars().count();

    if lower.contains("どれか")
        || lower.contains("どれですか")
        || lower.contains("選択して")
        || lower.contains("選んで")
        || lower.contains("正しいもの")
        || lower.contains("適切なもの")
        || char_cnt >= 45
    {
        QueryIntent::QuestionAnswering
    } else if lower.contains("違い")
        || lower.contains("比較")
        || lower.contains("メリット")
        || lower.contains("デメリット")
        || lower.contains("利点")
        || lower.contains("使い分け")
        || lower.contains("vs")
    {
        QueryIntent::ComparisonChoice
    } else if lower.contains("とは")
        || lower.contains("定義")
        || lower.contains("概要")
        || lower.contains("目的")
        || lower.contains("役割")
    {
        QueryIntent::DefinitionLookup
    } else if lower.contains("上限")
        || lower.contains("制限")
        || lower.contains("条件")
        || lower.contains("できない")
        || lower.contains("エラー")
        || lower.contains("設定")
    {
        QueryIntent::ConditionConstraint
    } else {
        QueryIntent::GeneralLookup
    }
}

/// Result of multi-feature relevance judgment on a candidate page.
pub struct PageJudgment {
    pub final_score: f32,
    pub confidence_pct: u8,
    pub rank: JudgmentRank,
    pub matched_terms: Vec<String>,
    pub missing_terms: Vec<String>,
    pub reasons: Vec<String>,
    pub best_sentence: String,
}

/// Evaluate a candidate page using the multi-feature judgment model:
/// 1. BM25 lexical relevance
/// 2. IDF-weighted keyword coverage
/// 3. Intra-sentence / window proximity (co-occurrence in the same sentence)
/// 4. Page title match boost
/// 5. Compound term exact match boost
/// 6. Query intent vs. Page role alignment
pub fn judge_page_relevance(
    page: &PageDocument,
    bm25_score: f32,
    max_bm25: f32,
    query_keywords: &[WeightedKeyword],
    intent: QueryIntent,
) -> PageJudgment {
    let search_lower = page.search_text.to_lowercase();
    let title_lower = page.page_title.nfkc().collect::<String>().to_lowercase();

    let mut matched_terms = Vec::new();
    let mut missing_terms = Vec::new();
    let mut matched_idf_sum = 0.0_f32;
    let mut total_idf_sum = 0.0_f32;
    let mut title_hits = 0_usize;
    let mut compound_hits = 0_usize;

    for kw in query_keywords {
        let lower = kw.term.nfkc().collect::<String>().to_lowercase();
        total_idf_sum += kw.idf;
        if search_lower.contains(&lower) {
            matched_idf_sum += kw.idf;
            matched_terms.push(kw.term.clone());
            if title_lower.contains(&lower) {
                title_hits += 1;
            }
            if kw.is_compound {
                compound_hits += 1;
            }
        } else {
            missing_terms.push(kw.term.clone());
        }
    }

    let idf_coverage = if total_idf_sum > 0.0 {
        (matched_idf_sum / total_idf_sum).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Sentence-level analysis: find the single best evidence sentence/segment on the page
    let (best_sentence, max_sent_terms, proximity_bonus) =
        extract_best_evidence_sentence(page, query_keywords);

    // Role alignment bonus
    let role_bonus: f32 = match (intent, page.page_role) {
        (_, PageRole::TitleCover) => -0.12,
        (QueryIntent::DefinitionLookup, PageRole::Definition | PageRole::Summary) => 0.14,
        (QueryIntent::ComparisonChoice, PageRole::ComparisonTable | PageRole::Summary) => 0.14,
        (QueryIntent::QuestionAnswering, PageRole::Definition | PageRole::Summary) => 0.12,
        (QueryIntent::QuestionAnswering, PageRole::QuizCheck) => 0.08,
        (_, PageRole::Definition | PageRole::Summary) => 0.06,
        _ => 0.0,
    };

    let norm_bm25 = if max_bm25 > 0.0 {
        (bm25_score / max_bm25).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let title_bonus = if title_hits > 0 {
        (0.15 + 0.05 * (title_hits as f32 - 1.0)).min(0.25)
    } else {
        0.0
    };

    let compound_bonus = (compound_hits as f32 * 0.08).min(0.20);

    // Final composite score
    let composite = (norm_bm25 * 0.35
        + idf_coverage * 0.40
        + proximity_bonus * 0.15
        + title_bonus
        + compound_bonus
        + role_bonus)
        .clamp(0.05, 1.25);

    let final_score = bm25_score * (0.5 + composite);
    let confidence_pct = ((composite / 1.05).clamp(0.12, 0.99) * 100.0).round() as u8;

    let rank = if confidence_pct >= 82 && (idf_coverage >= 0.70 || title_hits > 0) {
        JudgmentRank::S
    } else if confidence_pct >= 64 {
        JudgmentRank::A
    } else if confidence_pct >= 45 {
        JudgmentRank::B
    } else {
        JudgmentRank::C
    };

    let mut reasons = Vec::new();
    if !query_keywords.is_empty() {
        reasons.push(format!(
            "重要語カバー {}/{} ({:.0}%)",
            matched_terms.len(),
            query_keywords.len(),
            idf_coverage * 100.0
        ));
    }
    if title_hits > 0 {
        reasons.push("見出し一致".to_string());
    }
    if compound_hits > 0 {
        reasons.push("複合語一致".to_string());
    }
    if max_sent_terms >= 2 {
        reasons.push(format!("同一文に{}語近接", max_sent_terms));
    }
    reasons.push(format!("種別:{}", page.page_role.label()));

    PageJudgment {
        final_score,
        confidence_pct,
        rank,
        matched_terms,
        missing_terms,
        reasons,
        best_sentence,
    }
}

/// Split page into candidate sentences/blocks and score each by IDF-weighted term density
/// to extract the single most informative evidence sentence.
fn extract_best_evidence_sentence(
    page: &PageDocument,
    query_keywords: &[WeightedKeyword],
) -> (String, usize, f32) {
    let lines: Vec<&str> = page
        .display_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty()
                && !l.contains("Amazon Web Services, Inc.")
                && !l.contains("All rights reserved")
        })
        .collect();

    if lines.is_empty() {
        return (String::new(), 0, 0.0);
    }

    // Build sliding 2-line / sentence windows so bullet items and wrapped sentences stay together
    let mut segments: Vec<String> = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx + 1 < lines.len() {
            segments.push(format!("{} {}", line, lines[idx + 1]));
        }
        segments.push((*line).to_string());
    }

    let kw_lowers: Vec<(String, f32)> = query_keywords
        .iter()
        .map(|k| (k.term.nfkc().collect::<String>().to_lowercase(), k.idf))
        .collect();

    let mut best_seg = lines[0].to_string();
    let mut best_score = -1.0_f32;
    let mut best_term_count = 0_usize;

    for seg in &segments {
        let seg_lower = seg.nfkc().collect::<String>().to_lowercase();
        let mut hit_count = 0_usize;
        let mut idf_sum = 0.0_f32;

        for (kw_l, idf) in &kw_lowers {
            if seg_lower.contains(kw_l) {
                hit_count += 1;
                idf_sum += *idf;
            }
        }

        // Reward segments that contain multiple distinct query terms in compact length
        let len_penalty = (seg.chars().count() as f32 / 120.0).max(0.8);
        let score = (idf_sum * (1.0 + 0.35 * hit_count as f32)) / len_penalty;

        if hit_count > 0 && score > best_score {
            best_score = score;
            best_term_count = hit_count;
            best_seg = seg.clone();
        }
    }

    let total_kws = query_keywords.len().max(1);
    let proximity_bonus = if total_kws >= 2 {
        (best_term_count as f32 / total_kws as f32).clamp(0.0, 1.0)
    } else {
        best_term_count.min(1) as f32
    };

    // Trim best sentence to a readable length
    let chars: Vec<char> = best_seg.chars().collect();
    if chars.len() > 135 {
        let truncated: String = chars[..135].iter().collect();
        (format!("{}...", truncated), best_term_count, proximity_bonus)
    } else {
        (best_seg, best_term_count, proximity_bonus)
    }
}
