use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui::text::{CCursor, CCursorRange, LayoutJob};
use regex::Regex;

use crate::models::{
    AppSettings, JudgmentRank, PageDocument, PageRole, PdfFileInfo, QueryIntent, SearchMode,
    SearchResult, WeightedKeyword, DEFAULT_PDF_DIR,
};
use crate::pdf_loader::{
    cache_index_dir, compute_file_signatures, extract_all_pdfs, folder_cache_dir, is_cache_valid,
    load_app_settings, load_corpus_cache, save_app_settings, save_corpus_cache, scan_pdf_files,
};
use crate::search_engine::{build_highlight_regex, SearchEngine};

enum BgMessage {
    Progress {
        current: usize,
        total: usize,
        message: String,
    },
    Finished {
        pages: Vec<PageDocument>,
        files: Vec<PdfFileInfo>,
        engine: Box<SearchEngine>,
        from_cache: bool,
        elapsed_ms: u128,
    },
    Error(String),
}

pub struct AwsPdfSearchApp {
    pdf_dir: PathBuf,
    base_cache_dir: PathBuf,
    settings: AppSettings,

    // Indexing / loading state
    is_indexing: bool,
    progress_current: usize,
    progress_total: usize,
    progress_message: String,
    status_message: String,
    error_message: Option<String>,
    bg_rx: Option<Receiver<BgMessage>>,

    // Loaded corpus & engine
    pages: Vec<PageDocument>,
    files: Vec<PdfFileInfo>,
    enabled_files: HashSet<u32>,
    file_hit_counts: HashMap<u32, usize>,
    engine: Option<SearchEngine>,

    // Search state
    search_mode: SearchMode,
    query: String,
    multiline_input: bool,
    use_or: bool,
    extracted_keywords: Vec<WeightedKeyword>,
    excluded_keywords: HashSet<String>,
    extra_keywords: Vec<String>,
    related_suggestions: Vec<String>,
    detected_intent: QueryIntent,
    use_regex: bool,
    regex_error: Option<String>,

    // Debounce
    pending_search_at: Option<Instant>,
    last_analyzed_query: String,

    // Results & preview
    results: Vec<SearchResult>,
    highlight_terms: Vec<String>,
    highlight_regex: Option<Regex>,
    search_time_ms: f64,
    selected_result_idx: Option<usize>,
    current_doc_id: Option<usize>,
    scroll_to_selected: bool,
    show_inspector: bool,
    copied_toast_until: Option<Instant>,

    // Focus management
    request_focus_search: bool,
}

impl AwsPdfSearchApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_japanese_fonts(&cc.egui_ctx);
        configure_visuals(&cc.egui_ctx);

        let base_cache_dir = resolve_base_cache_dir();
        let mut settings = load_app_settings(&base_cache_dir);

        let default_dir = PathBuf::from(DEFAULT_PDF_DIR);
        let pdf_dir = settings
            .last_pdf_dir
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .unwrap_or(default_dir);

        record_recent_dir(&mut settings, &pdf_dir);
        save_app_settings(&base_cache_dir, &settings);

        let mut app = Self {
            pdf_dir,
            base_cache_dir,
            settings,
            is_indexing: false,
            progress_current: 0,
            progress_total: 0,
            progress_message: String::new(),
            status_message: "初期化中...".to_string(),
            error_message: None,
            bg_rx: None,
            pages: Vec::new(),
            files: Vec::new(),
            enabled_files: HashSet::new(),
            file_hit_counts: HashMap::new(),
            engine: None,
            search_mode: SearchMode::SmartAnalysis,
            query: String::new(),
            multiline_input: false,
            use_or: false,
            extracted_keywords: Vec::new(),
            excluded_keywords: HashSet::new(),
            extra_keywords: Vec::new(),
            related_suggestions: Vec::new(),
            detected_intent: QueryIntent::GeneralLookup,
            use_regex: false,
            regex_error: None,
            pending_search_at: None,
            last_analyzed_query: String::new(),
            results: Vec::new(),
            highlight_terms: Vec::new(),
            highlight_regex: None,
            search_time_ms: 0.0,
            selected_result_idx: None,
            current_doc_id: None,
            scroll_to_selected: false,
            show_inspector: true,
            copied_toast_until: None,
            request_focus_search: true,
        };

        app.start_load_or_index(&cc.egui_ctx, false);
        app
    }

    fn switch_folder(&mut self, ctx: &egui::Context, new_dir: PathBuf) {
        let target = if new_dir.is_file() {
            new_dir.parent().unwrap_or(&new_dir).to_path_buf()
        } else {
            new_dir
        };
        self.pdf_dir = target.clone();
        record_recent_dir(&mut self.settings, &target);
        save_app_settings(&self.base_cache_dir, &self.settings);
        self.start_load_or_index(ctx, false);
    }

    fn start_load_or_index(&mut self, ctx: &egui::Context, force_rebuild: bool) {
        self.is_indexing = true;
        self.progress_current = 0;
        self.progress_total = 0;
        self.progress_message = "PDFフォルダを再帰スキャン中...".to_string();
        self.error_message = None;

        let pdf_dir = self.pdf_dir.clone();
        let corpus_cache = folder_cache_dir(&self.base_cache_dir, &pdf_dir);
        let ctx_clone = ctx.clone();
        let (tx, rx) = mpsc::channel();
        self.bg_rx = Some(rx);

        std::thread::spawn(move || {
            let started = Instant::now();
            let pdfs = match scan_pdf_files(&pdf_dir) {
                Ok(p) if p.is_empty() => {
                    let _ = tx.send(BgMessage::Error(format!(
                        "PDFファイルが見つかりませんでした: {}",
                        pdf_dir.display()
                    )));
                    ctx_clone.request_repaint();
                    return;
                }
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(BgMessage::Error(e));
                    ctx_clone.request_repaint();
                    return;
                }
            };

            let sigs = compute_file_signatures(&pdf_dir, &pdfs);
            let index_dir = cache_index_dir(&corpus_cache);

            if !force_rebuild && is_cache_valid(&corpus_cache, &pdf_dir, &sigs) {
                let _ = tx.send(BgMessage::Progress {
                    current: 1,
                    total: 2,
                    message: "キャッシュからインデックスと解析モデルを読み込み中...".to_string(),
                });
                ctx_clone.request_repaint();

                if let Ok((pages, files, model)) = load_corpus_cache(&corpus_cache) {
                    if let Ok(engine) = SearchEngine::open_existing(&index_dir, model) {
                        let _ = tx.send(BgMessage::Finished {
                            pages,
                            files,
                            engine: Box::new(engine),
                            from_cache: true,
                            elapsed_ms: started.elapsed().as_millis(),
                        });
                        ctx_clone.request_repaint();
                        return;
                    }
                }
            }

            let total = pdfs.len();
            let _ = tx.send(BgMessage::Progress {
                current: 0,
                total,
                message: format!(
                    "全PDFの構造解析・テキスト抽出を実行中 (0 / {} ファイル)...",
                    total
                ),
            });
            ctx_clone.request_repaint();

            let tx_prog = tx.clone();
            let ctx_prog = ctx_clone.clone();
            let extracted = extract_all_pdfs(&pdf_dir, &pdfs, move |done, tot, fname| {
                let _ = tx_prog.send(BgMessage::Progress {
                    current: done,
                    total: tot,
                    message: format!("PDF構造解析中: {} / {} ({})", done, tot, fname),
                });
                ctx_prog.request_repaint();
            });

            let (mut pages, files) = match extracted {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(BgMessage::Error(e));
                    ctx_clone.request_repaint();
                    return;
                }
            };

            let _ = tx.send(BgMessage::Progress {
                current: total,
                total,
                message: format!(
                    "軽量解析モデル(TF-IDF・共起グラフ)学習＆インデックス構築中 ({} ページ)...",
                    pages.len()
                ),
            });
            ctx_clone.request_repaint();

            let engine = match SearchEngine::build_and_save(&index_dir, &mut pages) {
                Ok(eng) => eng,
                Err(e) => {
                    let _ = tx.send(BgMessage::Error(e));
                    ctx_clone.request_repaint();
                    return;
                }
            };

            if let Err(e) = save_corpus_cache(
                &corpus_cache,
                &pdf_dir,
                &sigs,
                &pages,
                &files,
                &engine.analysis_model,
            ) {
                let _ = tx.send(BgMessage::Error(format!(
                    "キャッシュ保存に失敗しましたが検索は可能です: {}",
                    e
                )));
            }

            let _ = tx.send(BgMessage::Finished {
                pages,
                files,
                engine: Box::new(engine),
                from_cache: false,
                elapsed_ms: started.elapsed().as_millis(),
            });
            ctx_clone.request_repaint();
        });
    }

    fn poll_background(&mut self) {
        let mut finished = false;
        if let Some(rx) = &self.bg_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    BgMessage::Progress {
                        current,
                        total,
                        message,
                    } => {
                        self.progress_current = current;
                        self.progress_total = total;
                        self.progress_message = message;
                    }
                    BgMessage::Finished {
                        pages,
                        files,
                        engine,
                        from_cache,
                        elapsed_ms,
                    } => {
                        let file_count = files.len();
                        let page_count = pages.len();
                        let vocab_size = engine.analysis_model.idf_map.len();
                        self.enabled_files = files.iter().map(|f| f.file_id).collect();
                        self.pages = pages;
                        self.files = files;
                        self.engine = Some(*engine);
                        self.is_indexing = false;
                        self.last_analyzed_query.clear();
                        self.status_message = if from_cache {
                            format!(
                                "{}ファイル / {}ページ / 語彙{}語 (キャッシュ {}ms)",
                                file_count, page_count, vocab_size, elapsed_ms
                            )
                        } else {
                            format!(
                                "{}ファイル / {}ページ / 語彙{}語 (全解析 {}ms)",
                                file_count, page_count, vocab_size, elapsed_ms
                            )
                        };
                        self.results.clear();
                        self.file_hit_counts.clear();
                        self.selected_result_idx = None;
                        self.current_doc_id = self.pages.first().map(|p| p.doc_id);
                        finished = true;
                    }
                    BgMessage::Error(err) => {
                        self.is_indexing = false;
                        self.error_message = Some(err);
                        self.status_message = "エラーが発生しました".to_string();
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.bg_rx = None;
            if !self.query.trim().is_empty() {
                self.run_search_now();
            }
        }
    }

    fn schedule_search(&mut self, ctx: &egui::Context) {
        // Auto-expand input box if user pasted a multi-line or long question
        if self.query.contains('\n') || self.query.chars().count() > 75 {
            self.multiline_input = true;
        }
        self.pending_search_at = Some(Instant::now() + Duration::from_millis(90));
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    fn run_search_now(&mut self) {
        self.pending_search_at = None;
        self.regex_error = None;

        let Some(engine) = &self.engine else {
            return;
        };

        let trimmed_q = self.query.trim().to_string();
        if !trimmed_q.is_empty() && trimmed_q.chars().count() >= 2 {
            record_recent_query(&mut self.settings, &trimmed_q);
            save_app_settings(&self.base_cache_dir, &self.settings);
        }

        let started = Instant::now();
        match self.search_mode {
            SearchMode::SmartAnalysis => {
                if self.query != self.last_analyzed_query {
                    self.extracted_keywords = engine.analyze_query_keywords(&self.query);
                    self.excluded_keywords.clear();
                    self.extra_keywords.clear();
                    self.last_analyzed_query = self.query.clone();
                }

                let mut active: Vec<WeightedKeyword> = self
                    .extracted_keywords
                    .iter()
                    .filter(|k| !self.excluded_keywords.contains(&k.term))
                    .cloned()
                    .collect();

                for extra in &self.extra_keywords {
                    if !active.iter().any(|k| k.term.eq_ignore_ascii_case(extra)) {
                        active.push(WeightedKeyword {
                            term: extra.clone(),
                            idf: engine.analysis_model.term_idf(extra),
                            is_compound: false,
                        });
                    }
                }

                let active_terms_for_cooc: Vec<String> =
                    active.iter().map(|k| k.term.clone()).collect();
                self.related_suggestions = engine
                    .analysis_model
                    .recommend_related_terms(&active_terms_for_cooc, 6);

                let (res, terms, intent) = engine.search_smart_analysis(
                    &self.query,
                    &active,
                    &self.enabled_files,
                    &self.pages,
                    150,
                );
                self.detected_intent = intent;
                self.highlight_regex = build_highlight_regex(&terms, None);
                self.highlight_terms = terms;
                self.results = res;
            }
            SearchMode::Normal => {
                let (res, terms) = engine.search_normal(
                    &self.query,
                    self.use_or,
                    &self.enabled_files,
                    &self.pages,
                    150,
                );
                self.related_suggestions =
                    engine.analysis_model.recommend_related_terms(&terms, 6);
                self.highlight_regex = build_highlight_regex(&terms, None);
                self.highlight_terms = terms;
                self.results = res;
            }
            SearchMode::SubstringRegex => {
                self.related_suggestions.clear();
                match engine.search_substring_regex(
                    &self.query,
                    self.use_regex,
                    &self.enabled_files,
                    &self.pages,
                    150,
                ) {
                    Ok((res, terms)) => {
                        self.highlight_regex = if self.use_regex {
                            build_highlight_regex(&[], terms.first().map(|s| s.as_str()))
                        } else {
                            build_highlight_regex(&terms, None)
                        };
                        self.highlight_terms = terms;
                        self.results = res;
                    }
                    Err(err) => {
                        self.regex_error = Some(err);
                        self.results.clear();
                        self.highlight_terms.clear();
                        self.highlight_regex = None;
                    }
                }
            }
        }

        self.search_time_ms = started.elapsed().as_secs_f64() * 1000.0;

        // Compute per-file hit counts
        self.file_hit_counts.clear();
        for r in &self.results {
            *self.file_hit_counts.entry(r.file_id).or_insert(0) += 1;
        }

        if let Some(first) = self.results.first() {
            self.selected_result_idx = Some(0);
            self.current_doc_id = Some(first.doc_id);
        } else {
            self.selected_result_idx = None;
        }
    }

    fn handle_shortcuts_and_drops(&mut self, ctx: &egui::Context) {
        let mut focus_search = false;
        let mut clear_query = false;
        let mut move_delta: i32 = 0;
        let mut dropped_path: Option<PathBuf> = None;

        ctx.input(|i| {
            if let Some(file) = i.raw.dropped_files.first() {
                let p = file.path();
                if !p.as_os_str().is_empty() {
                    dropped_path = Some(p.to_path_buf());
                }
            }
            if (i.modifiers.ctrl || i.modifiers.command)
                && (i.key_pressed(egui::Key::K) || i.key_pressed(egui::Key::F))
            {
                focus_search = true;
            }
            if i.key_pressed(egui::Key::Escape) {
                clear_query = true;
            }
            if !self.multiline_input || !ctx.egui_wants_keyboard_input() {
                if i.key_pressed(egui::Key::ArrowDown) {
                    move_delta = 1;
                } else if i.key_pressed(egui::Key::ArrowUp) {
                    move_delta = -1;
                }
            }
        });

        if let Some(path) = dropped_path {
            self.switch_folder(ctx, path);
        }

        if focus_search {
            self.request_focus_search = true;
        }

        if clear_query {
            self.clear_all_search();
        }

        if move_delta != 0 && !self.results.is_empty() {
            let len = self.results.len();
            let next_idx = match self.selected_result_idx {
                Some(idx) => {
                    if move_delta > 0 {
                        (idx + 1).min(len - 1)
                    } else {
                        idx.saturating_sub(1)
                    }
                }
                None => 0,
            };
            self.selected_result_idx = Some(next_idx);
            self.current_doc_id = Some(self.results[next_idx].doc_id);
            self.scroll_to_selected = true;
        }
    }

    fn clear_all_search(&mut self) {
        self.query.clear();
        self.extracted_keywords.clear();
        self.excluded_keywords.clear();
        self.extra_keywords.clear();
        self.related_suggestions.clear();
        self.last_analyzed_query.clear();
        self.results.clear();
        self.file_hit_counts.clear();
        self.highlight_terms.clear();
        self.highlight_regex = None;
        self.regex_error = None;
        self.selected_result_idx = None;
        self.request_focus_search = true;
    }

    fn navigate_adjacent_page(&mut self, delta: i32) {
        let Some(doc_id) = self.current_doc_id else {
            return;
        };
        let Some(cur) = self.pages.get(doc_id) else {
            return;
        };
        let target_id = if delta < 0 {
            doc_id.checked_sub(1)
        } else {
            Some(doc_id + 1)
        };
        if let Some(tid) = target_id {
            if let Some(next_page) = self.pages.get(tid) {
                if next_page.file_id == cur.file_id {
                    self.current_doc_id = Some(tid);
                }
            }
        }
    }

    fn jump_to_page_in_current_file(&mut self, target_page_num: u32) {
        let Some(doc_id) = self.current_doc_id else {
            return;
        };
        let Some(cur) = self.pages.get(doc_id) else {
            return;
        };
        let fid = cur.file_id;
        if let Some(found) = self
            .pages
            .iter()
            .find(|p| p.file_id == fid && p.page_num == target_page_num)
        {
            self.current_doc_id = Some(found.doc_id);
        }
    }
}

impl eframe::App for AwsPdfSearchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        self.poll_background();

        if let Some(at) = self.pending_search_at {
            if Instant::now() >= at {
                self.run_search_now();
            } else {
                ctx.request_repaint_after(Duration::from_millis(45));
            }
        }

        self.handle_shortcuts_and_drops(&ctx);

        // Top Panel: Folder Toolbar, Search Mode, Search Box, Analysis Chips, File Filter
        egui::Panel::top("top_controls").show(ui, |ui| {
            ui.add_space(4.0);

            // Row 1: App Title, Folder selector, Recent folders, Re-index, Status
            ui.horizontal(|ui| {
                ui.heading("🧠 PDF 全文解析・高速検索");
                ui.separator();

                if ui
                    .add_enabled(!self.is_indexing, egui::Button::new("📁 フォルダ選択"))
                    .on_hover_text("任意のPDFフォルダを選択（ウィンドウへフォルダをドラッグ＆ドロップも可）")
                    .clicked()
                {
                    if let Some(folder) = rfd::FileDialog::new()
                        .set_directory(&self.pdf_dir)
                        .pick_folder()
                    {
                        self.switch_folder(&ctx, folder);
                    }
                }

                // Recent folders dropdown
                if !self.settings.recent_dirs.is_empty() {
                    let mut chosen_dir: Option<PathBuf> = None;
                    egui::ComboBox::from_id_salt("recent_dirs_combo")
                        .selected_text(
                            self.pdf_dir
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("最近のフォルダ"),
                        )
                        .width(175.0)
                        .show_ui(ui, |ui| {
                            for d in &self.settings.recent_dirs {
                                let p = PathBuf::from(d);
                                let is_cur = p == self.pdf_dir;
                                if ui.selectable_label(is_cur, d).clicked() && !is_cur {
                                    chosen_dir = Some(p);
                                }
                            }
                        });
                    if let Some(dir) = chosen_dir {
                        self.switch_folder(&ctx, dir);
                    }
                }

                if ui
                    .add_enabled(!self.is_indexing, egui::Button::new("🔄 再解析・再インデックス"))
                    .on_hover_text("フォルダ内の全PDFを再スキャンして解析モデルとインデックスを再構築します")
                    .clicked()
                {
                    self.start_load_or_index(&ctx, true);
                }

                ui.label(
                    egui::RichText::new(self.pdf_dir.to_string_lossy())
                        .small()
                        .color(egui::Color32::from_gray(150)),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(&self.status_message)
                            .small()
                            .color(egui::Color32::from_rgb(125, 215, 155)),
                    );
                });
            });

            if self.is_indexing {
                ui.add_space(4.0);
                let frac = if self.progress_total > 0 {
                    self.progress_current as f32 / self.progress_total as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .text(&self.progress_message)
                        .animate(true),
                );
            }

            if let Some(err) = &self.error_message {
                ui.add_space(2.0);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 110, 110),
                    format!("⚠ {}", err),
                );
            }

            ui.separator();

            // Row 2: Search Mode Tabs & Mode-specific Controls
            let mut immediate_search = false;
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("モード:").strong());
                if ui
                    .selectable_value(
                        &mut self.search_mode,
                        SearchMode::SmartAnalysis,
                        "🧠 スマート解析・判断検索 (推奨)",
                    )
                    .on_hover_text("単語・短文・問題文すべてに対応。複合名詞＋TF-IDF重要度＋6特徴量判断モデルで最適ページと根拠文を特定します")
                    .changed()
                {
                    immediate_search = true;
                    self.request_focus_search = true;
                }
                if ui
                    .selectable_value(&mut self.search_mode, SearchMode::Normal, "🔍 通常検索 (AND/OR)")
                    .on_hover_text("スペース区切りAND/OR、\"フレーズ\"完全一致、-除外ワード検索")
                    .changed()
                {
                    immediate_search = true;
                    self.request_focus_search = true;
                }
                if ui
                    .selectable_value(
                        &mut self.search_mode,
                        SearchMode::SubstringRegex,
                        "🧩 部分一致 / 正規表現",
                    )
                    .on_hover_text("形態素解析を通さず全ページ本文を直接高速スキャンします")
                    .changed()
                {
                    immediate_search = true;
                    self.request_focus_search = true;
                }

                ui.separator();

                if ui
                    .toggle_value(&mut self.multiline_input, "📝 複数行(問題文)入力")
                    .on_hover_text("長文や試験問題文を貼り付けやすい複数行テキストボックスに切り替えます")
                    .changed()
                {
                    self.request_focus_search = true;
                }

                match self.search_mode {
                    SearchMode::SmartAnalysis => {
                        if !self.query.trim().is_empty() {
                            ui.separator();
                            ui.label(
                                egui::RichText::new(format!(
                                    "🎯 判定意図: {}",
                                    self.detected_intent.label()
                                ))
                                .small()
                                .strong()
                                .color(egui::Color32::from_rgb(255, 205, 100)),
                            );
                        }
                    }
                    SearchMode::Normal => {
                        if ui.checkbox(&mut self.use_or, "OR検索").changed() {
                            immediate_search = true;
                        }
                    }
                    SearchMode::SubstringRegex => {
                        if ui.checkbox(&mut self.use_regex, "正規表現").changed() {
                            immediate_search = true;
                        }
                    }
                }

                // Recent queries dropdown on the right
                if !self.settings.recent_queries.is_empty() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut picked_q: Option<String> = None;
                        egui::ComboBox::from_id_salt("recent_queries_combo")
                            .selected_text("🕒 検索履歴")
                            .width(130.0)
                            .show_ui(ui, |ui| {
                                for rq in &self.settings.recent_queries {
                                    let preview: String = rq.chars().take(42).collect();
                                    if ui.selectable_label(false, preview).clicked() {
                                        picked_q = Some(rq.clone());
                                    }
                                }
                            });
                        if let Some(q) = picked_q {
                            self.query = q;
                            self.last_analyzed_query.clear();
                            immediate_search = true;
                        }
                    });
                }
            });

            ui.add_space(4.0);

            // Row 3: Unified Search Input Box (Single-line or Multi-line)
            let search_id = ui.make_persistent_id("main_search_input");
            let hint = match self.search_mode {
                SearchMode::SmartAnalysis => {
                    "キーワード、質問、または試験の問題文をそのまま入力・貼り付け (Ctrl+K フォーカス / Esc クリア)..."
                }
                SearchMode::Normal => {
                    "キーワード入力 (例: VPC サブネット -NAT, \"責任共有モデル\")..."
                }
                SearchMode::SubstringRegex => {
                    if self.use_regex {
                        "正規表現を入力 (例: S3.*Glacier|EBS)..."
                    } else {
                        "部分一致文字列を入力..."
                    }
                }
            };

            ui.horizontal_top(|ui| {
                let box_width = (ui.available_width() - 80.0).max(200.0);
                let resp = if self.multiline_input {
                    ui.add_sized(
                        [box_width, 66.0],
                        egui::TextEdit::multiline(&mut self.query)
                            .id(search_id)
                            .desired_rows(3)
                            .hint_text(hint),
                    )
                } else {
                    ui.add_sized(
                        [box_width, 28.0],
                        egui::TextEdit::singleline(&mut self.query)
                            .id(search_id)
                            .hint_text(hint),
                    )
                };

                if self.request_focus_search {
                    resp.request_focus();
                    if let Some(mut state) = egui::TextEdit::load_state(&ctx, search_id) {
                        let len = self.query.chars().count();
                        state.cursor.set_char_range(Some(CCursorRange::two(
                            CCursor::new(0),
                            CCursor::new(len),
                        )));
                        state.store(&ctx, search_id);
                    }
                    self.request_focus_search = false;
                }

                if resp.changed() {
                    self.schedule_search(&ctx);
                }

                if ui.button("✕ クリア").clicked() {
                    self.clear_all_search();
                }
            });

            // Row 3b: Extracted Keywords & IDF Weights + Co-occurrence Recommendations
            if self.search_mode == SearchMode::SmartAnalysis && !self.extracted_keywords.is_empty()
            {
                ui.add_space(3.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("解析キーワード(重要度順):")
                            .small()
                            .strong()
                            .color(egui::Color32::from_gray(175)),
                    );
                    let mut toggled: Option<String> = None;
                    for kw in &self.extracted_keywords {
                        let excluded = self.excluded_keywords.contains(&kw.term);
                        let badge = if kw.is_compound {
                            format!("★{} ({:.1})", kw.term, kw.idf)
                        } else {
                            format!("{} ({:.1})", kw.term, kw.idf)
                        };
                        let text = if excluded {
                            egui::RichText::new(format!("✕ {}", kw.term))
                                .small()
                                .strikethrough()
                                .color(egui::Color32::from_gray(115))
                        } else {
                            egui::RichText::new(badge).small().color(egui::Color32::WHITE)
                        };
                        let fill = if excluded {
                            egui::Color32::from_gray(42)
                        } else if kw.is_compound {
                            egui::Color32::from_rgb(45, 110, 165)
                        } else {
                            egui::Color32::from_rgb(38, 85, 135)
                        };
                        if ui
                            .add(egui::Button::new(text).fill(fill).corner_radius(10.0))
                            .on_hover_text(if excluded {
                                "クリックで検索対象に戻す"
                            } else {
                                "クリックで検索から除外（括弧内はコーパスIDF重要度、★は複合名詞）"
                            })
                            .clicked()
                        {
                            toggled = Some(kw.term.clone());
                        }
                    }

                    // Extra user-added co-occurrence terms
                    let mut remove_extra: Option<String> = None;
                    for ext in &self.extra_keywords {
                        let text = egui::RichText::new(format!("+ {} ✕", ext))
                            .small()
                            .color(egui::Color32::WHITE);
                        if ui
                            .add(
                                egui::Button::new(text)
                                    .fill(egui::Color32::from_rgb(35, 125, 95))
                                    .corner_radius(10.0),
                            )
                            .on_hover_text("クリックで追加キーワードを解除")
                            .clicked()
                        {
                            remove_extra = Some(ext.clone());
                        }
                    }

                    if let Some(kw) = toggled {
                        if !self.excluded_keywords.remove(&kw) {
                            self.excluded_keywords.insert(kw);
                        }
                        immediate_search = true;
                    }
                    if let Some(ext) = remove_extra {
                        self.extra_keywords.retain(|x| x != &ext);
                        immediate_search = true;
                    }
                    if !self.excluded_keywords.is_empty() || !self.extra_keywords.is_empty() {
                        if ui.small_button("リセット").clicked() {
                            self.excluded_keywords.clear();
                            self.extra_keywords.clear();
                            immediate_search = true;
                        }
                    }
                });
            }

            // Related Co-occurring Terms Recommendation Bar
            if !self.related_suggestions.is_empty() {
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("💡 共起・関連語を追加:")
                            .small()
                            .color(egui::Color32::from_rgb(135, 200, 165)),
                    );
                    let mut clicked_rel: Option<String> = None;
                    for rel in &self.related_suggestions {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(format!("+ {}", rel))
                                        .small()
                                        .color(egui::Color32::from_rgb(185, 235, 205)),
                                )
                                .fill(egui::Color32::from_rgb(28, 52, 42))
                                .corner_radius(8.0),
                            )
                            .on_hover_text("同じページで共起しやすい関連重要語を検索条件に追加します")
                            .clicked()
                        {
                            clicked_rel = Some(rel.clone());
                        }
                    }
                    if let Some(rel) = clicked_rel {
                        if self.search_mode == SearchMode::SmartAnalysis {
                            if !self.extra_keywords.contains(&rel) {
                                self.extra_keywords.push(rel);
                            }
                        } else {
                            self.query = format!("{} {}", self.query.trim(), rel);
                        }
                        immediate_search = true;
                    }
                });
            }

            // Row 4: PDF File Filter Bar with Live Per-File Hit Counts
            if !self.files.is_empty() {
                ui.add_space(3.0);
                egui::ScrollArea::vertical()
                    .id_salt("files_filter_scroll")
                    .max_height(56.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "対象PDF ({}/{}):",
                                    self.enabled_files.len(),
                                    self.files.len()
                                ))
                                .small()
                                .strong(),
                            );
                            if ui.small_button("全選択").clicked() {
                                self.enabled_files =
                                    self.files.iter().map(|f| f.file_id).collect();
                                immediate_search = true;
                            }
                            if ui.small_button("全解除").clicked() {
                                self.enabled_files.clear();
                                immediate_search = true;
                            }
                            if !self.file_hit_counts.is_empty()
                                && ui
                                    .small_button("ヒットありのみ")
                                    .on_hover_text("現在の検索結果が含まれるPDFファイルだけに絞り込みます")
                                    .clicked()
                            {
                                self.enabled_files =
                                    self.file_hit_counts.keys().copied().collect();
                                immediate_search = true;
                            }
                            ui.separator();

                            for f in &self.files {
                                let mut checked = self.enabled_files.contains(&f.file_id);
                                let hit_cnt = self.file_hit_counts.get(&f.file_id).copied();
                                let label_text = match hit_cnt {
                                    Some(c) if c > 0 => format!("{} ({})", f.file_label, c),
                                    _ => f.file_label.clone(),
                                };
                                let rich = if hit_cnt.unwrap_or(0) > 0 {
                                    egui::RichText::new(label_text)
                                        .small()
                                        .strong()
                                        .color(egui::Color32::from_rgb(135, 210, 255))
                                } else {
                                    egui::RichText::new(label_text).small()
                                };

                                if ui
                                    .checkbox(&mut checked, rich)
                                    .on_hover_text(format!(
                                        "{} ({} ページ)",
                                        f.file_name, f.page_count
                                    ))
                                    .changed()
                                {
                                    if checked {
                                        self.enabled_files.insert(f.file_id);
                                    } else {
                                        self.enabled_files.remove(&f.file_id);
                                    }
                                    immediate_search = true;
                                }
                            }
                        });
                    });
            }

            ui.add_space(3.0);

            if immediate_search {
                self.run_search_now();
            }
        });

        // Left Pane: Ranked Search Results with Judgment Badges & Best Evidence
        egui::Panel::left("results_pane")
            .resizable(true)
            .default_size(490.0)
            .size_range(320.0..=820.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("検索結果: {} 件", self.results.len()))
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!("({:.2} ms)", self.search_time_ms))
                            .small()
                            .color(egui::Color32::from_gray(150)),
                    );

                    if !self.results.is_empty() {
                        let s_cnt = self
                            .results
                            .iter()
                            .filter(|r| r.rank == JudgmentRank::S)
                            .count();
                        let a_cnt = self
                            .results
                            .iter()
                            .filter(|r| r.rank == JudgmentRank::A)
                            .count();
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(format!("判断ランク S:{} / A:{}", s_cnt, a_cnt))
                                    .small()
                                    .color(egui::Color32::from_rgb(245, 200, 95)),
                            );
                        });
                    }
                });

                if let Some(err) = &self.regex_error {
                    ui.colored_label(egui::Color32::from_rgb(255, 110, 110), err);
                }

                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("results_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.results.is_empty() {
                            ui.add_space(20.0);
                            let msg = if self.is_indexing {
                                "フォルダ内の全PDFを解析しています..."
                            } else if self.query.trim().is_empty() {
                                "キーワードや問題文を入力すると、軽量判断モデルが関連度ランク(S〜C)と最重要根拠文を抽出します。\n\n• 任意のフォルダをウィンドウにドラッグ＆ドロップして解析可能\n• ↑ / ↓ キーで検索結果を移動"
                            } else {
                                "一致するページが見つかりませんでした。\nキーワードを減らすか、部分一致モードをお試しください。"
                            };
                            ui.label(
                                egui::RichText::new(msg).color(egui::Color32::from_gray(145)),
                            );
                            return;
                        }

                        let mut clicked_idx: Option<usize> = None;
                        for (idx, res) in self.results.iter().enumerate() {
                            let is_selected = self.selected_result_idx == Some(idx);
                            let fill = if is_selected {
                                egui::Color32::from_rgb(40, 64, 96)
                            } else {
                                egui::Color32::from_rgb(29, 32, 38)
                            };
                            let stroke = if is_selected {
                                egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 170, 255))
                            } else {
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 55, 64))
                            };

                            let frame_resp = egui::Frame::new()
                                .fill(fill)
                                .stroke(stroke)
                                .corner_radius(6.0)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());

                                    // Header line: Rank badge + File label + Page num + Role badge
                                    ui.horizontal(|ui| {
                                        let (rank_bg, rank_fg) = rank_colors(res.rank);
                                        egui::Frame::new()
                                            .fill(rank_bg)
                                            .corner_radius(4.0)
                                            .inner_margin(egui::Margin::symmetric(6, 2))
                                            .show(ui, |ui| {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{} {}%",
                                                        res.rank.label(),
                                                        res.confidence_pct
                                                    ))
                                                    .small()
                                                    .strong()
                                                    .color(rank_fg),
                                                );
                                            });

                                        ui.label(
                                            egui::RichText::new(&res.file_label)
                                                .strong()
                                                .color(egui::Color32::from_rgb(115, 195, 255)),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("p.{}", res.page_num))
                                                .strong()
                                                .color(egui::Color32::from_rgb(255, 210, 110)),
                                        );

                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    egui::RichText::new(res.page_role.label())
                                                        .small()
                                                        .color(role_color(res.page_role)),
                                                );
                                            },
                                        );
                                    });

                                    // Auto-extracted Page Title
                                    if !res.page_title.is_empty() {
                                        ui.add_space(2.0);
                                        ui.label(
                                            egui::RichText::new(format!("📌 {}", res.page_title))
                                                .strong()
                                                .size(13.0)
                                                .color(egui::Color32::from_gray(240)),
                                        );
                                    }

                                    // Judgment reason tags
                                    if !res.judgment_reasons.is_empty() {
                                        ui.add_space(2.0);
                                        ui.label(
                                            egui::RichText::new(res.judgment_reasons.join("  •  "))
                                                .small()
                                                .color(egui::Color32::from_rgb(145, 200, 165)),
                                        );
                                    }

                                    // Highlighted evidence snippet
                                    ui.add_space(3.0);
                                    let job = build_highlighted_layout_job(
                                        &res.snippet,
                                        self.highlight_regex.as_ref(),
                                        12.5,
                                        egui::Color32::from_gray(215),
                                    );
                                    ui.label(job);
                                });

                            let card_resp = ui.interact(
                                frame_resp.response.rect,
                                ui.make_persistent_id(("result_card", idx)),
                                egui::Sense::click(),
                            );
                            if card_resp.clicked() {
                                clicked_idx = Some(idx);
                            }
                            if is_selected && self.scroll_to_selected {
                                frame_resp.response.scroll_to_me(Some(egui::Align::Center));
                                self.scroll_to_selected = false;
                            }
                            ui.add_space(5.0);
                        }

                        if let Some(idx) = clicked_idx {
                            self.selected_result_idx = Some(idx);
                            self.current_doc_id = Some(self.results[idx].doc_id);
                        }
                    });
            });

        // Right Pane: Page Inspector & Full Text Preview
        egui::CentralPanel::default().show(ui, |ui| {
            let Some(doc_id) = self.current_doc_id else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("左の検索結果からページを選択してください")
                            .color(egui::Color32::from_gray(145)),
                    );
                });
                return;
            };

            let Some(page) = self.pages.get(doc_id).cloned() else {
                return;
            };

            let total_in_file = self
                .files
                .iter()
                .find(|f| f.file_id == page.file_id)
                .map(|f| f.page_count)
                .unwrap_or(page.page_num);

            let has_prev = doc_id > 0
                && self
                    .pages
                    .get(doc_id - 1)
                    .map(|p| p.file_id == page.file_id)
                    .unwrap_or(false);
            let has_next = self
                .pages
                .get(doc_id + 1)
                .map(|p| p.file_id == page.file_id)
                .unwrap_or(false);

            // Top Toolbar of Right Pane
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(&page.file_label)
                        .size(16.0)
                        .strong()
                        .color(egui::Color32::from_rgb(125, 205, 255)),
                );

                ui.separator();

                if ui
                    .add_enabled(has_prev, egui::Button::new("◀ 前"))
                    .on_hover_text("前のページへ移動")
                    .clicked()
                {
                    self.navigate_adjacent_page(-1);
                }

                // Direct page jump DragValue
                let mut target_page = page.page_num;
                let drag = ui.add(
                    egui::DragValue::new(&mut target_page)
                        .range(1..=total_in_file.max(1))
                        .prefix("p. ")
                        .suffix(format!(" / {}", total_in_file)),
                );
                if drag.changed() && target_page != page.page_num {
                    self.jump_to_page_in_current_file(target_page);
                }

                if ui
                    .add_enabled(has_next, egui::Button::new("次 ▶"))
                    .on_hover_text("次のページへ移動")
                    .clicked()
                {
                    self.navigate_adjacent_page(1);
                }

                ui.separator();

                // Font zoom controls
                if ui.small_button("A-").on_hover_text("文字を小さく").clicked() {
                    self.settings.font_size = (self.settings.font_size - 1.0).max(11.0);
                    save_app_settings(&self.base_cache_dir, &self.settings);
                }
                ui.label(
                    egui::RichText::new(format!("{:.0}px", self.settings.font_size))
                        .small()
                        .color(egui::Color32::from_gray(155)),
                );
                if ui.small_button("A+").on_hover_text("文字を大きく").clicked() {
                    self.settings.font_size = (self.settings.font_size + 1.0).min(26.0);
                    save_app_settings(&self.base_cache_dir, &self.settings);
                }

                ui.separator();
                ui.toggle_value(&mut self.show_inspector, "🧠 解析パネル");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let open_btn = egui::Button::new(
                        egui::RichText::new("📄 このページをPDFで開く")
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(200, 105, 25));

                    if ui.add(open_btn).clicked() {
                        if let Err(e) = open_pdf_at_page(&page.file_path, page.page_num) {
                            self.error_message = Some(e);
                        }
                    }

                    let copy_label = if self
                        .copied_toast_until
                        .map(|t| Instant::now() < t)
                        .unwrap_or(false)
                    {
                        "✓ コピー完了"
                    } else {
                        "📋 全文コピー"
                    };
                    if ui.button(copy_label).clicked() {
                        ctx.copy_text(page.display_text.clone());
                        self.copied_toast_until =
                            Some(Instant::now() + Duration::from_millis(1500));
                    }
                });
            });

            // Subtitle row: filename, auto-extracted title, page role
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!("📌 {}", page.page_title))
                        .strong()
                        .color(egui::Color32::from_gray(235)),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("種別: {}", page.page_role.label()))
                        .small()
                        .color(role_color(page.page_role)),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(&page.file_name)
                        .small()
                        .color(egui::Color32::from_gray(140)),
                );
            });

            // Lightweight Analysis & Judgment Inspector Card
            if self.show_inspector {
                ui.add_space(4.0);
                let mut clicked_page_term: Option<String> = None;

                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(27, 33, 43))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 72, 95)))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());

                        // Show judgment details if the currently viewed page is in search results
                        if let Some(res) = self.results.iter().find(|r| r.doc_id == page.doc_id) {
                            ui.horizontal_wrapped(|ui| {
                                let (rank_bg, rank_fg) = rank_colors(res.rank);
                                egui::Frame::new()
                                    .fill(rank_bg)
                                    .corner_radius(4.0)
                                    .inner_margin(egui::Margin::symmetric(6, 2))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "判断ランク {} (信頼度 {}%)",
                                                res.rank.label(),
                                                res.confidence_pct
                                            ))
                                            .small()
                                            .strong()
                                            .color(rank_fg),
                                        );
                                    });

                                if !res.matched_terms.is_empty() {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "✓ 一致語: {}",
                                            res.matched_terms.join(", ")
                                        ))
                                        .small()
                                        .color(egui::Color32::from_rgb(130, 225, 160)),
                                    );
                                }
                                if !res.missing_terms.is_empty() {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "• 未一致: {}",
                                            res.missing_terms.join(", ")
                                        ))
                                        .small()
                                        .color(egui::Color32::from_gray(135)),
                                    );
                                }
                            });

                            if !res.best_sentence.is_empty() {
                                ui.add_space(3.0);
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(
                                        egui::RichText::new("🎯 最重要根拠:")
                                            .small()
                                            .strong()
                                            .color(egui::Color32::from_rgb(255, 205, 100)),
                                    );
                                    let job = build_highlighted_layout_job(
                                        &res.best_sentence,
                                        self.highlight_regex.as_ref(),
                                        12.5,
                                        egui::Color32::from_gray(230),
                                    );
                                    ui.label(job);
                                });
                            }
                            ui.add_space(3.0);
                        }

                        // Page's own top TF-IDF domain concepts (clickable to search!)
                        if !page.top_terms.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new("🏷 このページの特徴重要語(TF-IDF):")
                                        .small()
                                        .color(egui::Color32::from_gray(170)),
                                );
                                for (term, score) in &page.top_terms {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new(format!(
                                                    "{} ({:.1})",
                                                    term, score
                                                ))
                                                .small()
                                                .color(egui::Color32::from_rgb(190, 220, 255)),
                                            )
                                            .fill(egui::Color32::from_rgb(38, 52, 72))
                                            .corner_radius(8.0),
                                        )
                                        .on_hover_text("クリックでこの重要語を検索します")
                                        .clicked()
                                    {
                                        clicked_page_term = Some(term.clone());
                                    }
                                }
                            });
                        }
                    });

                if let Some(term) = clicked_page_term {
                    self.query = term;
                    self.last_analyzed_query.clear();
                    self.run_search_now();
                }
            }

            ui.separator();

            // Selectable & copyable full-page text with keyword highlighting
            let hl_re = self.highlight_regex.clone();
            let font_sz = self.settings.font_size;
            let mut preview_copy = page.display_text.clone();

            egui::ScrollArea::vertical()
                .id_salt(("preview_scroll", doc_id))
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut layouter =
                        move |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
                            let mut job = build_highlighted_layout_job(
                                buf.as_str(),
                                hl_re.as_ref(),
                                font_sz,
                                egui::Color32::from_gray(235),
                            );
                            job.wrap.max_width = wrap_width;
                            ui.fonts_mut(|f| f.layout_job(job))
                        };

                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut preview_copy)
                            .frame(egui::Frame::NONE)
                            .layouter(&mut layouter),
                    );
                });
        });
    }
}

fn rank_colors(rank: JudgmentRank) -> (egui::Color32, egui::Color32) {
    match rank {
        JudgmentRank::S => (
            egui::Color32::from_rgb(215, 160, 25),
            egui::Color32::from_rgb(20, 15, 0),
        ),
        JudgmentRank::A => (
            egui::Color32::from_rgb(40, 145, 85),
            egui::Color32::WHITE,
        ),
        JudgmentRank::B => (
            egui::Color32::from_rgb(50, 105, 175),
            egui::Color32::WHITE,
        ),
        JudgmentRank::C => (
            egui::Color32::from_gray(75),
            egui::Color32::from_gray(225),
        ),
    }
}

fn role_color(role: PageRole) -> egui::Color32 {
    match role {
        PageRole::Definition => egui::Color32::from_rgb(130, 210, 255),
        PageRole::Summary => egui::Color32::from_rgb(150, 235, 170),
        PageRole::ComparisonTable => egui::Color32::from_rgb(255, 195, 120),
        PageRole::DiagramSlide => egui::Color32::from_rgb(215, 165, 255),
        PageRole::QuizCheck => egui::Color32::from_rgb(255, 150, 160),
        PageRole::TitleCover => egui::Color32::from_gray(150),
    }
}

fn record_recent_dir(settings: &mut AppSettings, dir: &Path) {
    let s = dir.to_string_lossy().to_string();
    settings.last_pdf_dir = Some(s.clone());
    settings.recent_dirs.retain(|d| !d.eq_ignore_ascii_case(&s));
    settings.recent_dirs.insert(0, s);
    settings.recent_dirs.truncate(10);
}

fn record_recent_query(settings: &mut AppSettings, q: &str) {
    let s = q.trim().to_string();
    if s.is_empty() {
        return;
    }
    settings.recent_queries.retain(|x| x != &s);
    settings.recent_queries.insert(0, s);
    settings.recent_queries.truncate(12);
}

fn build_highlighted_layout_job(
    text: &str,
    re: Option<&Regex>,
    font_size: f32,
    base_color: egui::Color32,
) -> LayoutJob {
    let mut job = LayoutJob::default();
    let normal_fmt = egui::TextFormat {
        font_id: egui::FontId::proportional(font_size),
        color: base_color,
        ..Default::default()
    };
    let highlight_fmt = egui::TextFormat {
        font_id: egui::FontId::proportional(font_size),
        color: egui::Color32::BLACK,
        background: egui::Color32::from_rgb(255, 215, 70),
        ..Default::default()
    };

    let Some(regex) = re else {
        job.append(text, 0.0, normal_fmt);
        return job;
    };

    let mut last_end = 0;
    for m in regex.find_iter(text) {
        if m.start() > last_end {
            job.append(&text[last_end..m.start()], 0.0, normal_fmt.clone());
        }
        if m.end() > m.start() {
            job.append(&text[m.start()..m.end()], 0.0, highlight_fmt.clone());
        }
        last_end = m.end();
    }
    if last_end < text.len() {
        job.append(&text[last_end..], 0.0, normal_fmt);
    }
    job
}

fn configure_japanese_fonts(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
        r"C:\Windows\Fonts\BIZ-UDGothicR.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ];

    let mut fonts = egui::FontDefinitions::default();
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(
                "jp_system_font".to_owned(),
                Arc::new(egui::FontData::from_owned(bytes)),
            );
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "jp_system_font".to_owned());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .insert(0, "jp_system_font".to_owned());
            break;
        }
    }
    ctx.set_fonts(fonts);
}

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgb(22, 25, 31);
    visuals.extreme_bg_color = egui::Color32::from_rgb(16, 18, 23);
    ctx.set_visuals(visuals);
}

fn resolve_base_cache_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.join("cache");
        }
    }
    PathBuf::from("cache")
}

fn open_pdf_at_page(file_path: &str, page_num: u32) -> Result<(), String> {
    let url = build_pdf_file_url(file_path, page_num);

    let edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "msedge",
    ];

    for edge in edge_candidates {
        if edge == "msedge" || Path::new(edge).exists() {
            if Command::new(edge).arg(&url).spawn().is_ok() {
                return Ok(());
            }
        }
    }

    Command::new("cmd")
        .args(["/C", "start", "", &url])
        .spawn()
        .map_err(|e| format!("PDFをブラウザで開けませんでした: {}", e))?;
    Ok(())
}

fn build_pdf_file_url(file_path: &str, page_num: u32) -> String {
    let normalized = file_path.replace('\\', "/");
    let mut encoded = String::with_capacity(normalized.len() * 2);
    for b in normalized.bytes() {
        let keep = b.is_ascii_alphanumeric()
            || matches!(b, b'/' | b':' | b'-' | b'_' | b'.' | b'~');
        if keep {
            encoded.push(b as char);
        } else {
            encoded.push_str(&format!("%{:02X}", b));
        }
    }
    format!("file:///{}#page={}", encoded, page_num)
}
