//! HTTPS-only text retrieval with a single, validated and pinned destination.

use std::net::{Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::process::Command;

const MAX_BYTES: usize = 256 * 1024;

fn allowed_ip(ip: Ipv4Addr) -> bool {
    !matches!(
        ip.octets(),
        [0 | 10 | 127 | 224..=255, _, _, _]
            | [100, 64..=127, _, _]
            | [169, 254, _, _]
            | [172, 16..=31, _, _]
            | [192, 0, 0 | 2, _]
            | [192, 88, 99, _]
            | [192, 168, _, _]
            | [198, 18 | 19, _, _]
            | [198, 51, 100, _]
            | [203, 0, 113, _]
    )
}

pub fn host_for_url(url: &str) -> Result<&str, &'static str> {
    let rest = url
        .strip_prefix("https://")
        .ok_or("only HTTPS is enabled")?;
    if url.len() > 2048
        || !url.bytes().all(|b| b.is_ascii_graphic() && b != b'\\')
        || url.contains('#')
    {
        return Err("invalid HTTPS URL");
    }
    let host = rest.split(['/', '?']).next().unwrap_or_default();
    let valid = host.len() <= 253
        && host.rsplit_once('.').is_some_and(|(_, tld)| {
            tld.bytes().any(|b| b.is_ascii_alphabetic())
                && !tld.eq_ignore_ascii_case("local")
                && !tld.eq_ignore_ascii_case("localhost")
                && !tld.eq_ignore_ascii_case("internal")
        })
        && host.split('.').all(|part| {
            !part.is_empty()
                && part.len() <= 63
                && !part.starts_with('-')
                && !part.ends_with('-')
                && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        });
    valid
        .then_some(host)
        .ok_or("a public DNS name on HTTPS port 443 is required")
}

pub fn fetch_public_https(url: &str) -> Result<Vec<u8>, (bool, &'static str)> {
    let host = host_for_url(url).map_err(|e| (true, e))?;
    let resolved = (host, 443)
        .to_socket_addrs()
        .map_err(|_| (false, "DNS lookup failed"))?;
    let mut ipv4 = None;
    for addr in resolved {
        if let SocketAddr::V4(v4) = addr {
            if !allowed_ip(*v4.ip()) {
                return Err((
                    true,
                    "DNS returned an address outside the public IPv4 policy",
                ));
            }
            ipv4 = Some(*v4.ip());
        }
    }
    let ip = ipv4.ok_or((true, "no public IPv4 address was found"))?;
    let pin = format!("{host}:443:{ip}");
    let out = Command::new("curl")
        .args([
            "--disable",
            "--silent",
            "--show-error",
            "--fail",
            "--globoff",
            "--ipv4",
            "--noproxy",
            "*",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--resolve",
            &pin,
            "--max-filesize",
            "262144",
            "--connect-timeout",
            "5",
            "--max-time",
            "15",
            "--write-out",
            "\n%{http_code}",
            "--",
            url,
        ])
        .output()
        .map_err(|_| (false, "HTTPS client unavailable"))?;
    let Some(separator) = out.stdout.iter().rposition(|&b| b == b'\n') else {
        return Err((false, "HTTPS retrieval failed"));
    };
    let (body, suffix) = out.stdout.split_at(separator);
    let code = &suffix[1..];
    if !out.status.success() || body.len() > MAX_BYTES || code.len() != 3 || code[0] != b'2' {
        return Err((false, "HTTPS retrieval failed or the page redirected"));
    }
    Ok(body.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_browser_policy() {
        assert!(host_for_url("https://example.com/example").is_ok());
        assert!(host_for_url("http://example.com/").is_err());
        assert!(host_for_url("https://localhost/").is_err());
        assert!(host_for_url("https://example.com:444/").is_err());
        assert!(host_for_url("https://user@example.com/").is_err());
        assert!(allowed_ip(Ipv4Addr::new(1, 1, 1, 1)));
        assert!(!allowed_ip(Ipv4Addr::LOCALHOST));
        assert!(!allowed_ip(Ipv4Addr::new(100, 64, 0, 1)));
    }
}
