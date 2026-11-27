//! HTTPS document retrieval with a freshly validated, pinned destination at each hop.

use std::borrow::Cow;
use std::net::{Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::process::Command;

use url::Url;

const MAX_BYTES: usize = 256 * 1024;
type FetchError = (bool, &'static str);

pub struct Page {
    pub url: String,
    pub bytes: Vec<u8>,
    pub html: bool,
}

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

pub fn fetch_public_https(url: &str) -> Result<Page, FetchError> {
    let mut url = url.to_owned();
    for _ in 0..4 {
        let (page, redirect) = fetch_once(&url)?;
        match redirect {
            Some(next) => url = next,
            None => return Ok(page),
        }
    }
    Err((true, "HTTPS redirect limit reached"))
}

fn fetch_once(url: &str) -> Result<(Page, Option<String>), FetchError> {
    let host = host_for_url(url).map_err(|e| (true, e))?;
    // Use the absolute DNS name so a resolver search suffix cannot change the target.
    let resolved = (format!("{host}."), 443)
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
            "8",
            "--write-out",
            "\n%{http_code}\t%{content_type}\t%{redirect_url}",
            "--",
            url,
        ])
        .output()
        .map_err(|_| (false, "HTTPS client unavailable"))?;
    let failed = (false, "HTTPS retrieval failed");
    let separator = out.stdout.iter().rposition(|&b| b == b'\n').ok_or(failed)?;
    let (body, suffix) = out.stdout.split_at(separator);
    let mut metadata = std::str::from_utf8(&suffix[1..])
        .map_err(|_| failed)?
        .splitn(3, '\t');
    let code: u16 = metadata.next().ok_or(failed)?.parse().map_err(|_| failed)?;
    let content_type = metadata.next().ok_or(failed)?;
    let redirect = metadata.next().ok_or(failed)?;
    if !out.status.success() || body.len() > MAX_BYTES {
        return Err(failed);
    }
    let mut page = Page {
        url: url.to_owned(),
        bytes: Vec::new(),
        html: false,
    };
    if matches!(code, 301 | 302 | 303 | 307 | 308) && !redirect.is_empty() {
        let mut target = Url::parse(redirect).map_err(|_| (true, "invalid HTTPS redirect"))?;
        target.set_fragment(None);
        host_for_url(target.as_str()).map_err(|error| (true, error))?;
        return Ok((page, Some(target.into())));
    }
    if !(200..300).contains(&code) {
        return Err(failed);
    }
    let mime = content_type.split(';').next().unwrap_or_default().trim();
    page.html = mime.eq_ignore_ascii_case("text/html")
        || mime.eq_ignore_ascii_case("application/xhtml+xml");
    if !page.html && !mime.eq_ignore_ascii_case("text/plain") {
        return Err((true, "this browser supports HTML and plain text documents"));
    }
    page.bytes = body.to_vec();
    Ok((page, None))
}

pub fn render_document(url: &str, raw: &str) -> String {
    let Ok(base) = Url::parse(url) else {
        return String::new();
    };
    ammonia::Builder::empty()
        .add_url_schemes(["https"])
        .add_tags([
            "article",
            "section",
            "main",
            "header",
            "footer",
            "nav",
            "aside",
            "div",
            "span",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "br",
            "hr",
            "a",
            "ul",
            "ol",
            "li",
            "dl",
            "dt",
            "dd",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
            "pre",
            "code",
            "blockquote",
            "strong",
            "em",
            "b",
            "i",
            "sup",
            "sub",
            "details",
            "summary",
        ])
        .add_clean_content_tags([
            "head", "script", "style", "iframe", "noscript", "template", "svg", "math",
        ])
        .add_tag_attributes("a", ["href"])
        .attribute_filter(move |_, attr, value| {
            if attr != "href" {
                return None;
            }
            let mut target = base.join(value).ok()?;
            target.set_fragment(None);
            host_for_url(target.as_str()).ok()?;
            let encoded: String =
                url::form_urlencoded::byte_serialize(target.as_str().as_bytes()).collect();
            Some(Cow::Owned(format!("#inkling-page-{encoded}")))
        })
        .clean(raw)
        .to_string()
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

    #[test]
    fn rendered_links_stay_in_the_broker_and_active_content_is_removed() {
        let page = render_document(
            "https://example.com/docs/",
            r#"<div onclick="bad()"><a href="../more?q=1">more</a><a href="https://example.com/next">next</a><a href="//localhost/">bad link</a><img src="//external.test/pixel"><script>bad()</script></div>"#,
        );
        assert!(
            page.contains("href=\"#inkling-page-https%3A%2F%2Fexample.com%2Fmore%3Fq%3D1\""),
            "{page}"
        );
        assert!(!page.contains("onclick"));
        assert!(page.contains("href=\"#inkling-page-https%3A%2F%2Fexample.com%2Fnext\""));
        assert!(!page.contains("localhost"));
        assert!(!page.contains("pixel"));
        assert!(!page.contains("bad()"));
    }
}
