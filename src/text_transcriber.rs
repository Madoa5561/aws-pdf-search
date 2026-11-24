use pdf_oxide::extractors::text::{ArtifactType, PaginationSubtype};
use pdf_oxide::layout::TextSpan;
use pdf_oxide::PdfDocument;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone)]
struct SpanItem {
    text: String,
    x: f32,
    y: f32,
    width: f32,
    font_size: f32,
}

/// A contiguous horizontal segment on a single line within one column/box.
#[derive(Debug, Clone)]
struct LineSegment {
    text: String,
    x_left: f32,
    x_right: f32,
    y_center: f32,
    font_size: f32,
}

/// A 2D text block (paragraph, bullet list, or diagram box) composed of vertically stacked LineSegments.
#[derive(Debug, Clone)]
struct TextBoxBlock {
    lines: Vec<LineSegment>,
    x_left: f32,
    x_right: f32,
    y_top: f32,
    y_bottom: f32,
}

/// Extract and reconstruct clean, natural Japanese & English text from a PDF page
/// using 2D span geometry, column-aware text box clustering, and linguistic line-wrap reconstruction.
pub fn transcribe_page(doc: &PdfDocument, page_index: usize) -> String {
    let page_num = (page_index + 1) as u32;
    if let Ok(spans) = doc.extract_spans(page_index) {
        if !spans.is_empty() {
            let reconstructed = transcribe_from_spans(&spans, page_num);
            if !reconstructed.trim().is_empty() {
                return reconstructed;
            }
        }
    }
    let raw = doc.extract_text(page_index).unwrap_or_default();
    reconstruct_from_raw_lines(&raw, page_num)
}

fn transcribe_from_spans(spans: &[TextSpan], page_num: u32) -> String {
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for s in spans {
        if s.text.trim().is_empty() {
            continue;
        }
        min_y = min_y.min(s.bbox.y);
        max_y = max_y.max(s.bbox.y + s.bbox.height);
    }
    let page_h = (max_y - min_y).max(100.0);

    let mut items: Vec<SpanItem> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(at) = &s.artifact_type {
            match at {
                ArtifactType::Pagination(PaginationSubtype::PageNumber)
                | ArtifactType::Pagination(PaginationSubtype::Footer)
                | ArtifactType::Pagination(PaginationSubtype::Header) => continue,
                _ => {}
            }
        }

        let cleaned = s.text.replace('\r', "").replace('\n', " ");
        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.contains("Amazon Web Services, Inc. or its affiliates")
            || trimmed.contains("All rights reserved")
        {
            continue;
        }

        let is_bottom_region = (s.bbox.y - min_y) <= page_h * 0.14;
        if is_bottom_region
            && (trimmed == page_num.to_string()
                || (trimmed.len() <= 3 && trimmed.chars().all(|c| c.is_ascii_digit())))
        {
            continue;
        }

        let fs = s.font_size.max(8.0);
        items.push(SpanItem {
            text: cleaned,
            x: s.bbox.x,
            y: s.bbox.y,
            width: s.bbox.width.max(0.0),
            font_size: fs,
        });
    }

    if items.is_empty() {
        return String::new();
    }

    // Sort top-to-bottom (Y descending in PDF user space), then left-to-right (X ascending)
    items.sort_by(|a, b| {
        b.y.partial_cmp(&a.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 1. Cluster spans into horizontal rows by Y coordinate
    let mut y_rows: Vec<Vec<SpanItem>> = Vec::new();
    for item in items {
        let mut placed = false;
        for row in y_rows.iter_mut().rev().take(5) {
            let ref_y = row[0].y;
            let ref_fs = row[0].font_size.max(item.font_size);
            if (ref_y - item.y).abs() <= ref_fs * 0.52 {
                row.push(item.clone());
                placed = true;
                break;
            }
        }
        if !placed {
            y_rows.push(vec![item]);
        }
    }

    y_rows.sort_by(|a, b| {
        b[0].y
            .partial_cmp(&a[0].y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 2. Within each Y-row, split at wide horizontal gaps into column-local LineSegments
    let mut segments: Vec<LineSegment> = Vec::new();
    for mut row_spans in y_rows {
        row_spans.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let mut cur_group: Vec<SpanItem> = Vec::new();
        for sp in row_spans {
            if let Some(prev) = cur_group.last() {
                let prev_right = prev.x + prev.width;
                let gap = sp.x - prev_right;
                let fs = prev.font_size.max(sp.font_size);

                // Skip duplicate overlapping glyph runs
                if gap < -fs * 0.45 && sp.text.trim() == prev.text.trim() {
                    continue;
                }

                // If there is a wide horizontal gap (> 2.2 * font_size) and prev is not a bullet marker,
                // start a new column segment on the same row!
                let prev_is_bullet = is_standalone_bullet(prev.text.trim());
                if !prev_is_bullet && gap > fs * 2.2 {
                    if let Some(seg) = build_line_segment(&cur_group) {
                        segments.push(seg);
                    }
                    cur_group.clear();
                }
            }
            cur_group.push(sp);
        }
        if let Some(seg) = build_line_segment(&cur_group) {
            segments.push(seg);
        }
    }

    // 3. First merge standalone bullet segments ("•") with the segment immediately to their right or below
    let mut merged_bullets: Vec<LineSegment> = Vec::with_capacity(segments.len());
    let mut used = vec![false; segments.len()];
    for i in 0..segments.len() {
        if used[i] {
            continue;
        }
        if is_standalone_bullet(&segments[i].text) {
            // Find the nearest segment on the same line (to the right) or immediately below
            let b = &segments[i];
            let mut best_j: Option<usize> = None;
            let mut best_dist = f32::MAX;
            for j in (i + 1)..segments.len() {
                if used[j] || is_standalone_bullet(&segments[j].text) {
                    continue;
                }
                let cand = &segments[j];
                let dy = (b.y_center - cand.y_center).abs();
                let dx = (cand.x_left - b.x_left).max(0.0);
                if dy <= cand.font_size * 1.4 && dx <= cand.font_size * 6.0 {
                    let dist = dy * 2.0 + dx;
                    if dist < best_dist {
                        best_dist = dist;
                        best_j = Some(j);
                    }
                }
            }
            if let Some(j) = best_j {
                used[j] = true;
                let mut target = segments[j].clone();
                target.text = format!(
                    "• {}",
                    target
                        .text
                        .trim_start_matches(|c: char| is_bullet_char(c) || c.is_whitespace())
                );
                target.x_left = b.x_left.min(target.x_left);
                merged_bullets.push(target);
                continue;
            }
        }
        merged_bullets.push(segments[i].clone());
    }

    // 4. Cluster LineSegments into 2D TextBoxBlocks (vertically stacked lines in the same X column)
    let mut blocks: Vec<TextBoxBlock> = Vec::new();
    for seg in merged_bullets {
        let mut attached = false;
        // Search existing blocks from most recent to see if `seg` sits directly below one in the same column
        for blk in blocks.iter_mut().rev() {
            let last_line = blk.lines.last().unwrap();
            let fs = last_line.font_size.max(seg.font_size);
            let dy = last_line.y_center - seg.y_center;

            // Must be directly below within ~1.85x line height and similar font size
            if !(0.0..=fs * 1.85).contains(&dy) || (last_line.font_size - seg.font_size).abs() > 2.2
            {
                continue;
            }

            // Check horizontal column affinity: either aligned left edge OR strong horizontal overlap within the same column width
            let left_diff = (seg.x_left - blk.x_left).abs();
            let overlap = (blk.x_right.min(seg.x_right) - blk.x_left.max(seg.x_left)).max(0.0);
            let min_w = (blk.x_right - blk.x_left)
                .min(seg.x_right - seg.x_left)
                .max(fs);

            let same_column = left_diff <= fs * 2.4 || (overlap / min_w) >= 0.45;
            if same_column {
                blk.x_left = blk.x_left.min(seg.x_left);
                blk.x_right = blk.x_right.max(seg.x_right);
                blk.y_bottom = blk.y_bottom.min(seg.y_center);
                blk.lines.push(seg.clone());
                attached = true;
                break;
            }
        }

        if !attached {
            blocks.push(TextBoxBlock {
                x_left: seg.x_left,
                x_right: seg.x_right,
                y_top: seg.y_center,
                y_bottom: seg.y_center,
                lines: vec![seg],
            });
        }
    }

    // Order blocks in natural reading order: top-to-bottom by y_top (with row band tolerance), then left-to-right by x_left
    blocks.sort_by(|a, b| {
        let fs = a
            .lines
            .first()
            .map(|l| l.font_size)
            .unwrap_or(12.0)
            .max(b.lines.first().map(|l| l.font_size).unwrap_or(12.0));
        if (a.y_top - b.y_top).abs() <= fs * 0.85 {
            a.x_left
                .partial_cmp(&b.x_left)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            b.y_top
                .partial_cmp(&a.y_top)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    // 5. Reconstruct paragraphs inside each 2D TextBoxBlock
    let mut rendered_blocks: Vec<String> = Vec::with_capacity(blocks.len());
    for (blk_idx, blk) in blocks.iter().enumerate() {
        let block_text = format_block_lines(&blk.lines, blk_idx == 0, page_num);
        if !block_text.is_empty() && block_text != page_num.to_string() {
            rendered_blocks.push(block_text);
        }
    }

    rendered_blocks.join("\n\n").trim().to_string()
}

fn build_line_segment(spans: &[SpanItem]) -> Option<LineSegment> {
    if spans.is_empty() {
        return None;
    }
    let x_left = spans.first().map(|s| s.x).unwrap_or(0.0);
    let mut x_right = x_left;
    let mut y_sum = 0.0;
    let mut max_fs = 0.0_f32;
    let mut line_str = String::new();

    for (idx, sp) in spans.iter().enumerate() {
        y_sum += sp.y;
        max_fs = max_fs.max(sp.font_size);
        let sp_right = sp.x + sp.width;

        if idx == 0 {
            line_str.push_str(sp.text.trim());
        } else {
            let prev = &spans[idx - 1];
            let prev_right = prev.x + prev.width;
            let gap = sp.x - prev_right;
            let fs = prev.font_size.max(sp.font_size);

            let prev_c = line_str.chars().last();
            let next_c = sp.text.trim().chars().next();

            if needs_space_between_spans(prev_c, next_c, gap, fs, &prev.text, &sp.text) {
                line_str.push(' ');
            }
            line_str.push_str(sp.text.trim());
        }
        x_right = x_right.max(sp_right);
    }

    let cleaned = clean_inline_japanese_spacing(&line_str);
    if cleaned.is_empty() {
        None
    } else {
        Some(LineSegment {
            text: cleaned,
            x_left,
            x_right,
            y_center: y_sum / (spans.len() as f32),
            font_size: max_fs,
        })
    }
}

/// Merge soft-wrapped lines inside a single 2D TextBoxBlock into clean sentences/bullets.
fn format_block_lines(lines: &[LineSegment], is_top_block: bool, page_num: u32) -> String {
    if lines.is_empty() {
        return String::new();
    }

    let mut merged: Vec<LineSegment> = Vec::with_capacity(lines.len());
    for (idx, line) in lines.iter().cloned().enumerate() {
        if merged.is_empty() {
            merged.push(line);
            continue;
        }

        let prev = merged.last().unwrap();
        if should_merge_in_block(prev, &line, is_top_block && idx == 1) {
            let prev_mut = merged.last_mut().unwrap();
            if needs_space_on_line_wrap(&prev_mut.text, &line.text) {
                prev_mut.text.push(' ');
            }
            prev_mut.text.push_str(line.text.trim());
            prev_mut.text = clean_inline_japanese_spacing(&prev_mut.text);
            prev_mut.y_center = line.y_center;
            prev_mut.x_right = prev_mut.x_right.max(line.x_right);
        } else {
            merged.push(line);
        }
    }

    let page_suffix = page_num.to_string();
    let mut out_lines: Vec<String> = Vec::with_capacity(merged.len());
    for (idx, l) in merged.into_iter().enumerate() {
        let mut t = l.text.trim().to_string();
        if idx + 1 == lines.len() && t.ends_with(&page_suffix) && t.len() > page_suffix.len() {
            let prefix = &t[..t.len() - page_suffix.len()];
            if prefix.ends_with('。') || prefix.ends_with(')') || prefix.ends_with('）') {
                t = prefix.to_string();
            }
        }
        if !t.is_empty() {
            out_lines.push(t);
        }
    }

    out_lines.join("\n")
}

/// Because `prev` and `next` are ALREADY verified to be inside the same 2D TextBoxBlock
/// (same column, close vertical distance, similar font size), they should merge whenever
/// `next` is a wrapped continuation rather than a new bullet or new sentence.
fn should_merge_in_block(prev: &LineSegment, next: &LineSegment, prev_is_slide_title: bool) -> bool {
    let p_txt = prev.text.trim();
    let n_txt = next.text.trim();
    if p_txt.is_empty() || n_txt.is_empty() {
        return false;
    }

    // Broken parenthetical expressions like "(\nプライベートクラウド\n)"
    if p_txt.ends_with('(')
        || p_txt.ends_with('（')
        || p_txt.ends_with('「')
        || n_txt.starts_with(')')
        || n_txt.starts_with('）')
        || n_txt.starts_with('」')
        || n_txt.starts_with('、')
        || n_txt.starts_with('。')
    {
        return true;
    }

    // "モジュール\n1: クラウドのコンセプト"
    if (p_txt == "モジュール" || p_txt.eq_ignore_ascii_case("module"))
        && n_txt.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
    {
        return true;
    }

    // Never merge if `next` starts a new bullet item or list item
    if starts_with_bullet_or_list_marker(n_txt) {
        return false;
    }

    // Never merge if `prev` ends with sentence-final punctuation
    if p_txt.ends_with('。')
        || p_txt.ends_with('！')
        || p_txt.ends_with('？')
        || p_txt.ends_with('!')
        || p_txt.ends_with('?')
    {
        return false;
    }

    if ends_with_continuation_cue(p_txt) || starts_with_continuation_cue(n_txt) {
        return true;
    }

    // If `prev` is the top slide title and didn't end with a continuation cue, keep subtitle on next line
    if prev_is_slide_title {
        return false;
    }

    // Inside the same text box, if `prev` has at least 6 chars (e.g. "予測に基づくデータセ" + "ンターへの投資",
    // "使用した量に対" + "してのみ支払う"), merge wrapped lines!
    let p_chars = p_txt.chars().count();
    if p_chars >= 6 {
        return true;
    }

    false
}

fn needs_space_between_spans(
    prev_c: Option<char>,
    next_c: Option<char>,
    gap: f32,
    font_size: f32,
    prev_raw: &str,
    next_raw: &str,
) -> bool {
    let (Some(p), Some(n)) = (prev_c, next_c) else {
        return false;
    };

    if is_open_bracket(p) || is_close_bracket_or_punct(n) || p == '、' || p == '。' || p == '・' {
        return false;
    }

    if is_bullet_char(p) {
        return true;
    }

    if p.is_ascii_graphic() && n.is_ascii_graphic() {
        if prev_raw.ends_with(' ') || next_raw.starts_with(' ') || gap > font_size * 0.14 {
            return true;
        }
        return false;
    }

    if is_cjk_or_kana(p) && is_cjk_or_kana(n) {
        if is_katakana(p)
            && is_katakana(n)
            && (prev_raw.ends_with(' ') || next_raw.starts_with(' ') || gap > font_size * 0.35)
        {
            return true;
        }
        return false;
    }

    if p == ':' || p == '：' {
        return true;
    }

    false
}

fn ends_with_continuation_cue(s: &str) -> bool {
    let trimmed = s.trim_end();
    if trimmed.ends_with('、')
        || trimmed.ends_with('ー')
        || trimmed.ends_with(',')
        || trimmed.ends_with('(')
        || trimmed.ends_with('（')
    {
        return true;
    }
    for suffix in [
        "は", "が", "を", "に", "へ", "と", "で", "や", "の", "から", "まで", "より", "および",
        "または", "かつ", "および他の", "主要な", "さまざまな", "以下の", "次の", "対",
    ] {
        if trimmed.ends_with(suffix) && trimmed.chars().count() >= 3 {
            return true;
        }
    }
    false
}

fn starts_with_continuation_cue(s: &str) -> bool {
    let trimmed = s.trim_start();
    for prefix in [
        "、", "。", "）", ")", "」", "で、", "を、", "に、", "は、", "が、", "の", "を", "に",
        "へ", "と", "で", "や", "から", "まで", "です", "ます", "して", "ンター", "ビス",
    ] {
        if trimmed.starts_with(prefix) {
            return true;
        }
    }
    false
}

fn needs_space_on_line_wrap(prev_text: &str, next_text: &str) -> bool {
    let p = prev_text.trim_end().chars().last();
    let n = next_text.trim_start().chars().next();
    match (p, n) {
        (Some(pc), Some(nc)) => {
            if is_open_bracket(pc) || is_close_bracket_or_punct(nc) {
                false
            } else if pc.is_ascii_alphanumeric() && nc.is_ascii_alphanumeric() {
                true
            } else {
                pc == ':'
            }
        }
        _ => false,
    }
}

pub fn reconstruct_from_raw_lines(raw: &str, page_num: u32) -> String {
    let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<String> = Vec::new();
    let raw_lines: Vec<&str> = normalized.lines().map(|l| l.trim()).collect();

    let mut i = 0;
    while i < raw_lines.len() {
        let line = raw_lines[i];
        if line.is_empty()
            || line.contains("Amazon Web Services, Inc. or its affiliates")
            || line == page_num.to_string()
        {
            i += 1;
            continue;
        }
        if is_standalone_bullet(line) && i + 1 < raw_lines.len() && !raw_lines[i + 1].is_empty() {
            lines.push(format!("• {}", clean_inline_japanese_spacing(raw_lines[i + 1])));
            i += 2;
            continue;
        }
        lines.push(clean_inline_japanese_spacing(line));
        i += 1;
    }

    lines.join("\n").trim().to_string()
}

pub fn clean_inline_japanese_spacing(input: &str) -> String {
    let nfkc_spaces = input.replace('\u{00A0}', " ").replace('\u{3000}', " ");
    let collapsed: String = nfkc_spaces.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = collapsed.chars().collect();
    let mut out = String::with_capacity(collapsed.len());

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' {
            let prev = out.chars().last();
            let next = chars.get(i + 1).copied();
            if should_drop_space(prev, next) {
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }

    if out.starts_with('•') && !out.starts_with("• ") {
        out = format!("• {}", out.trim_start_matches('•').trim_start());
    }
    out.trim().to_string()
}

fn should_drop_space(prev: Option<char>, next: Option<char>) -> bool {
    let (Some(p), Some(n)) = (prev, next) else {
        return true;
    };
    if is_bullet_char(p) {
        return false;
    }
    if is_open_bracket(p) || is_close_bracket_or_punct(n) || p == '、' || p == '。' {
        return true;
    }
    if (is_cjk_or_kana(p) && is_hiragana(n)) || (is_hiragana(p) && is_cjk_or_kana(n)) {
        return true;
    }
    if (p.is_ascii_alphanumeric() && is_cjk_or_kana(n))
        || (is_cjk_or_kana(p) && n.is_ascii_alphanumeric())
    {
        return true;
    }
    false
}

fn is_bullet_char(c: char) -> bool {
    matches!(c, '•' | '・' | '▪' | '▸' | '◦' | '➢')
}

fn is_standalone_bullet(s: &str) -> bool {
    let t = s.trim();
    t == "•" || t == "・" || t == "-" || t == "▪" || t == "▸"
}

fn starts_with_bullet_or_list_marker(s: &str) -> bool {
    let t = s.trim_start();
    t.starts_with('•')
        || t.starts_with('・')
        || t.starts_with("- ")
        || t.starts_with("* ")
        || t.starts_with('▪')
        || t.starts_with('▸')
}

fn is_open_bracket(c: char) -> bool {
    matches!(c, '(' | '（' | '[' | '［' | '「' | '『' | '【' | '〈' | '《')
}

fn is_close_bracket_or_punct(c: char) -> bool {
    matches!(
        c,
        ')' | '）' | ']' | '］' | '」' | '』' | '】' | '〉' | '》' | '、' | '。' | '，' | '．' | '！' | '？'
    )
}

fn is_hiragana(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

fn is_katakana(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c) || ('\u{FF65}'..='\u{FF9F}').contains(&c)
}

fn is_cjk_or_kana(c: char) -> bool {
    is_hiragana(c)
        || is_katakana(c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c)
        || ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

pub fn build_search_text_from_display(
    display_text: &str,
    boilerplate: &std::collections::HashSet<String>,
) -> String {
    let filtered_lines: Vec<&str> = display_text
        .lines()
        .filter(|l| {
            let key = crate::analyzer_model::normalize_boilerplate_key(l);
            !boilerplate.contains(&key)
        })
        .collect();
    let joined = filtered_lines.join("\n");
    let nfkc: String = joined.nfkc().collect();
    let chars: Vec<char> = nfkc.chars().collect();
    let mut out = String::with_capacity(nfkc.len());

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            let prev = out.chars().last();
            let next = chars.get(i + 1).copied();
            match (prev, next) {
                (Some(p), Some(n)) if is_cjk_or_kana(p) && is_cjk_or_kana(n) => {}
                (Some(p), Some(n))
                    if p == '、'
                        || p == '。'
                        || n == '、'
                        || n == '。'
                        || n == '）'
                        || p == '（' => {}
                _ => {
                    out.push(' ');
                }
            }
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DEFAULT_PDF_DIR;
    use std::path::PathBuf;

    #[test]
    fn test_transcribe_module01_pages() {
        let path = PathBuf::from(DEFAULT_PDF_DIR).join("AcademyCloudFoundations_Module_01.pdf");
        if !path.exists() {
            return;
        }
        let doc = PdfDocument::open(&path).expect("open module 01");
        for p in [0, 1, 2, 5, 10, 14, 19] {
            let txt = transcribe_page(&doc, p);
            println!("=== TRANSCRIBED PAGE {} ===\n{}\n", p + 1, txt);
            assert!(
                !txt.contains("データベー\nス"),
                "Broken word データベー\\nス on page {}",
                p + 1
            );
            assert!(
                !txt.contains("サー\nビス"),
                "Broken word サー\\nビス on page {}",
                p + 1
            );
        }
    }
}
