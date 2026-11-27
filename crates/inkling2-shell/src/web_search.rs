//! Search through an operator-selected local RustXNG instance.

use std::process::Command;

use serde_json::{Value, json};
use url::Url;

use crate::public_browser::host_for_url;

pub fn search(port: u16, query: &str) -> Result<Value, &'static str> {
    let mut url = Url::parse(&format!("http://127.0.0.1:{port}/search"))
        .map_err(|_| "invalid search port")?;
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("format", "json")
        .append_pair("max_results", "12");
    let response = Command::new("curl")
        .args([
            "--disable",
            "--silent",
            "--fail",
            "--globoff",
            "--noproxy",
            "*",
            "--proto",
            "=http",
            "--connect-timeout",
            "2",
            "--max-time",
            "25",
            "--max-filesize",
            "262144",
            "--header",
            "Accept: application/json",
            "--",
            url.as_str(),
        ])
        .output()
        .map_err(|_| "RustXNG client unavailable")?;
    if !response.status.success() || response.stdout.len() > 256 * 1024 {
        return Err("RustXNG did not return search results");
    }
    let raw: Value =
        serde_json::from_slice(&response.stdout).map_err(|_| "invalid RustXNG response")?;
    let rows = raw["results"].as_array().ok_or("invalid RustXNG results")?;
    let results: Vec<_> = rows
        .iter()
        .filter_map(|row| {
            let mut url = Url::parse(row["url"].as_str()?).ok()?;
            url.set_fragment(None);
            host_for_url(url.as_str()).ok()?;
            let title: String = row["title"]
                .as_str()
                .unwrap_or(url.as_str())
                .chars()
                .take(240)
                .collect();
            let snippet: String = row["content"]
                .as_str()
                .unwrap_or_default()
                .chars()
                .take(480)
                .collect();
            Some(json!({"url": url.as_str(), "title": title, "snippet": snippet}))
        })
        .take(12)
        .collect();
    Ok(json!({"results": results, "provider": "RustXNG"}))
}

pub fn valid_query(query: &str) -> bool {
    !query.trim().is_empty() && query.len() <= 512 && !query.chars().any(char::is_control)
}
