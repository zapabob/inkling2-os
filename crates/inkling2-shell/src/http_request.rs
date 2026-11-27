//! Request boundary for the loopback console and native-client API.

use std::io::{BufRead, Read};
use std::net::SocketAddr;

const MAX_HEAD_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 256 * 1024;

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    pub auth: Option<String>,
}

pub type Rejection = (&'static str, &'static str);

const BAD_HEADER: Rejection = ("400 Bad Request", "invalid request headers");
const BAD_HOST: Rejection = ("403 Forbidden", "untrusted Host or browser origin");
const INVALID_BODY: Rejection = ("400 Bad Request", "invalid request body");

pub fn bearer_matches(header: Option<&str>, token: Option<&str>) -> bool {
    let (Some(actual), Some(expected)) = (header.and_then(|h| h.strip_prefix("Bearer ")), token)
    else {
        return false;
    };
    actual.len() == expected.len()
        && actual
            .bytes()
            .zip(expected.bytes())
            .fold(0, |diff, (a, b)| diff | (a ^ b))
            == 0
}

pub fn read_request(reader: &mut impl BufRead, addr: SocketAddr) -> Result<Request, Rejection> {
    let mut sections = (&mut *reader).take(MAX_HEAD_BYTES as u64);
    let mut line = String::new();
    if sections.read_line(&mut line).is_err() || !line.ends_with("\r\n") {
        return Err(BAD_HEADER);
    }
    let mut parts = line.trim_end_matches("\r\n").split(' ');
    let method = parts.next().ok_or(BAD_HEADER)?.to_owned();
    let path = parts
        .next()
        .filter(|p| p.starts_with('/') && !p.starts_with("//"))
        .and_then(|p| p.split('?').next())
        .ok_or(BAD_HEADER)?
        .to_owned();
    if parts.next() != Some("HTTP/1.1") || parts.next().is_some() {
        return Err(BAD_HEADER);
    }

    let mut host = None;
    let mut origin = None;
    let mut auth = None;
    let mut fetch_site = None;
    let mut length = None;
    let mut content_type = None;
    loop {
        line.clear();
        if sections.read_line(&mut line).is_err() || !line.ends_with("\r\n") {
            return Err(BAD_HEADER);
        }
        if line == "\r\n" {
            break;
        }
        let (name, value) = line
            .trim_end_matches("\r\n")
            .split_once(':')
            .ok_or(BAD_HEADER)?;
        if !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
            || name.is_empty()
            || value.bytes().any(|c| c.is_ascii_control() && c != b'\t')
        {
            return Err(BAD_HEADER);
        }
        let slot = match name.to_ascii_lowercase().as_str() {
            "host" => &mut host,
            "origin" => &mut origin,
            "authorization" => &mut auth,
            "sec-fetch-site" => &mut fetch_site,
            "content-length" => &mut length,
            "content-type" => &mut content_type,
            "transfer-encoding" => return Err(BAD_HEADER),
            _ => continue,
        };
        if slot.replace(value.trim().to_owned()).is_some() {
            return Err(BAD_HEADER);
        }
    }

    let authority = addr.to_string();
    if host.as_deref() != Some(authority.as_str()) {
        return Err(BAD_HOST);
    }
    if origin.is_some_and(|o| o != format!("http://{authority}"))
        || fetch_site.is_some_and(|s| !matches!(s.as_str(), "same-origin" | "none"))
    {
        return Err(BAD_HOST);
    }
    let length = length
        .map_or(Ok(0), |value| value.parse::<usize>())
        .map_err(|_| BAD_HEADER)?;
    let limit = if method == "POST" && path == "/v1/chat/completions" {
        16 * 1024 * 1024
    } else {
        MAX_BODY_BYTES
    };
    if length > limit {
        return Err((
            "413 Payload Too Large",
            "request body exceeds the route limit",
        ));
    }
    if method == "POST"
        && !content_type.as_deref().is_some_and(|t| {
            t.split(';')
                .next()
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))
        })
    {
        return Err(("415 Unsupported Media Type", "expected application/json"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).map_err(|_| INVALID_BODY)?;
    Ok(Request {
        method,
        path,
        body,
        auth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_and_header_framing_are_checked() -> Result<(), Box<dyn std::error::Error>> {
        let addr = "127.0.0.1:8080".parse()?;
        let request = |head: &str| {
            let bytes = format!("GET /health HTTP/1.1\r\n{head}\r\n");
            read_request(&mut bytes.as_bytes(), addr)
        };
        let good = "Host: 127.0.0.1:8080\r\n";
        assert!(request(good).is_ok());
        assert_eq!(request("Host: invalid.example\r\n").err(), Some(BAD_HOST));
        assert_eq!(
            request(&format!("{good}Origin: https://invalid.example\r\n")).err(),
            Some(BAD_HOST)
        );
        assert_eq!(
            request(&format!("{good}Content-Length: 0\r\nContent-Length: 0\r\n")).err(),
            Some(BAD_HEADER)
        );
        assert_eq!(
            request(&format!("{good}Transfer-Encoding: chunked\r\n")).err(),
            Some(BAD_HEADER)
        );
        assert!(bearer_matches(Some("Bearer local"), Some("local")));
        assert!(!bearer_matches(Some("Bearer local"), Some("other")));
        Ok(())
    }
}
