#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analyzer_model;
mod app;
mod models;
mod pdf_loader;
mod search_engine;
mod text_transcriber;

use app::AwsPdfSearchApp;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("PDF 全文解析・高速検索ツール (軽量解析・判断モデル搭載 / 完全オフライン)")
            .with_inner_size([1420.0, 900.0])
            .with_min_inner_size([960.0, 620.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "PDF 全文解析・高速検索ツール",
        options,
        Box::new(|cc| Ok(Box::new(AwsPdfSearchApp::new(cc)))),
    )
}
