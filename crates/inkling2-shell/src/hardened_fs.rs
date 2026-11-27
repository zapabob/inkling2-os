//! Read-only previews confined to a directory capability chosen by the operator.

use std::io::Read;
use std::path::Path;

use cap_fs_ext::{DirExt, FollowSymlinks, MetadataExt, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};

const MAX_DIR_ENTRIES: usize = 200;
const MAX_READ_BYTES: usize = 64 * 1024;
const PREVIEW_EXTENSIONS: &[&str] = &["txt", "md", "rst", "csv"];

pub struct FsEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub blocked: bool,
}

pub struct FsReadResult {
    pub rel_path: String,
    pub size: u64,
    pub truncated: bool,
    pub fnv64_hex: String,
    pub content: String,
}

fn allowed_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let stem = lower.split('.').next().unwrap_or_default();
    !name.is_empty()
        && !name.starts_with('.')
        && !name.ends_with(['.', ' '])
        && !name
            .chars()
            .any(|c| c.is_control() || matches!(c, ':' | '\\' | '~'))
        && !matches!(stem, "con" | "nul" | "aux" | "prn")
        && !stem.starts_with("com")
        && !stem.starts_with("lpt")
        && ![
            "secret",
            "credential",
            "id_ed25519",
            "id_rsa",
            "api_key",
            "apikey",
            "password",
            "token",
        ]
        .iter()
        .any(|word| lower.contains(word))
        && ![".key", ".pem", ".pfx", ".p12"]
            .iter()
            .any(|ext| lower.ends_with(ext))
}

fn path_parts(rel: &str) -> Result<Vec<&str>, &'static str> {
    if rel.is_empty() {
        return Ok(Vec::new());
    }
    let parts: Vec<_> = rel.split('/').collect();
    if rel.len() > 4096 || parts.len() > 32 || parts.iter().any(|name| !allowed_name(name)) {
        return Err("path is outside the preview policy");
    }
    Ok(parts)
}

fn open_dir(root: &Dir, parts: &[&str]) -> Result<Dir, &'static str> {
    let mut dir = root.try_clone().map_err(|_| "root unavailable")?;
    for part in parts {
        dir = dir
            .open_dir_nofollow(part)
            .map_err(|_| "directory unavailable")?;
    }
    Ok(dir)
}

fn preview_extension(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            PREVIEW_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        })
}

pub fn list_jailed_dir(
    root: &Dir,
    rel: &str,
    previews: bool,
) -> Result<Vec<FsEntry>, &'static str> {
    let dir = open_dir(root, &path_parts(rel)?)?;
    let entries = dir.entries().map_err(|_| "directory unavailable")?;
    let mut result = Vec::new();
    for entry in entries.take(MAX_DIR_ENTRIES).flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if !allowed_name(&name) {
            continue;
        }
        let Ok(meta) = dir.symlink_metadata(&name) else {
            continue;
        };
        if !(meta.is_dir() || meta.is_file()) {
            continue;
        }
        result.push(FsEntry {
            blocked: meta.is_file() && !(previews && preview_extension(&name) && meta.nlink() == 1),
            name,
            is_dir: meta.is_dir(),
            size: meta.len(),
        });
    }
    result.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));
    Ok(result)
}

pub fn read_jailed_file(root: &Dir, rel: &str) -> Result<FsReadResult, &'static str> {
    let parts = path_parts(rel)?;
    let (name, parents) = parts.split_last().ok_or("select a text file")?;
    if !preview_extension(name) {
        return Err("file type is outside the preview policy");
    }
    let dir = open_dir(root, parents)?;
    let candidate = dir.symlink_metadata(name).map_err(|_| "file unavailable")?;
    if !candidate.is_file() || candidate.nlink() != 1 {
        return Err("only regular, singly linked files can be previewed");
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    #[cfg(unix)]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = dir
        .open_with(name, &options)
        .map_err(|_| "file unavailable")?;
    let opened = file.metadata().map_err(|_| "file unavailable")?;
    if !opened.is_file() || opened.nlink() != 1 {
        return Err("only regular, singly linked files can be previewed");
    }
    let mut bytes = Vec::new();
    file.take(MAX_READ_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "preview read failed")?;
    let truncated = bytes.len() > MAX_READ_BYTES;
    bytes.truncate(MAX_READ_BYTES);
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(err) if truncated && err.error_len().is_none() => {
            std::str::from_utf8(&bytes[..err.valid_up_to()]).map_err(|_| "text must be UTF-8")?
        }
        Err(_) => return Err("text must be UTF-8"),
    };
    if text
        .chars()
        .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
    {
        return Err("binary preview disabled");
    }
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    Ok(FsReadResult {
        rel_path: rel.to_owned(),
        size: opened.len(),
        truncated,
        fnv64_hex: format!("{hash:016x}"),
        content: text.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cap_std::ambient_authority;
    use std::fs;

    #[test]
    fn preview_requires_regular_public_text_beneath_explicit_root() -> std::io::Result<()> {
        let dir = std::env::temp_dir().join(format!("inkling2-preview-{}", std::process::id()));
        fs::create_dir(&dir)?;
        fs::write(dir.join("hello.txt"), "Inkling2-OS 日本語")?;
        fs::write(dir.join(".env"), "private")?;
        fs::hard_link(dir.join(".env"), dir.join("alias.txt"))?;
        #[cfg(unix)]
        std::os::unix::fs::symlink("hello.txt", dir.join("link.txt"))?;
        let root = Dir::open_ambient_dir(&dir, ambient_authority())?;
        assert!(read_jailed_file(&root, "hello.txt").is_ok_and(|r| r.content.ends_with("日本語")));
        for path in [
            "../hello.txt",
            "/hello.txt",
            ".env",
            "hello.txt:stream",
            "alias.txt",
            "link.txt",
        ] {
            assert!(read_jailed_file(&root, path).is_err(), "{path}");
        }
        drop(root);
        for name in ["hello.txt", ".env", "alias.txt"] {
            fs::remove_file(dir.join(name))?;
        }
        #[cfg(unix)]
        fs::remove_file(dir.join("link.txt"))?;
        fs::remove_dir(dir)
    }
}
