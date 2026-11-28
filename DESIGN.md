---
name: Inkling2-OS Strata macOS Console
version: 1.0.0
theme_modes:
  - light
  - dark
colors:
  light:
    canvas: "#f5f5f7"
    sidebar: "rgba(246, 246, 248, 0.78)"
    surface: "#ffffff"
    surface_subtle: "#f2f3f5"
    border: "rgba(0, 0, 0, 0.08)"
    text_primary: "#1d1d1f"
    text_secondary: "#515154"
    text_tertiary: "#6e6e73"
    accent: "#0071e3"
    accent_hover: "#0077ed"
    cuda_emerald: "#16a34a"
    cuda_cyan: "#0891b2"
    warning: "#d97706"
    danger: "#dc2626"
  dark:
    canvas: "#0e1113"
    sidebar: "rgba(22, 25, 29, 0.78)"
    surface: "#16191d"
    surface_subtle: "#1e2228"
    border: "rgba(255, 255, 255, 0.10)"
    text_primary: "#f5f5f7"
    text_secondary: "#b8bcc4"
    text_tertiary: "#8b929e"
    accent: "#2997ff"
    accent_hover: "#4aa8ff"
    cuda_emerald: "#22c55e"
    cuda_cyan: "#22d3ee"
    warning: "#f59e0b"
    danger: "#ef4444"
typography:
  sans: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', 'Hiragino Sans', 'Yu Gothic UI', sans-serif"
  display: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', 'Hiragino Sans', sans-serif"
  mono: "'SF Mono', ui-monospace, 'Cascadia Code', 'Consolas', monospace"
spacing:
  unit: 4px
  sidebar_width: 272px
  titlebar_height: 52px
rounded:
  window: 14px
  card: 12px
  control: 8px
  pill: 9999px
components:
  - macOS Titlebar with traffic-light indicators and segmented view switcher
  - Frosted Translucent Sidebar with live RTX 5060 Ti CUDA 13.5 telemetry
  - Task-first Chat & Reasoning Playground
  - CUDA 13.5 sm_120a Tensor Core Micro-Bench & Telemetry View
  - Memory45 (45 MiB) & Dynamic LoRA Inspector
  - SC117 .strata v1.1 Model & CLI Operations View
---

# Inkling2-OS — Strata-Aligned macOS Console Design System

## 1. Visual Philosophy
- **Product-first macOS desktop feel**: Frosted translucent sidebar (`backdrop-filter: blur(20px) saturate(160%)`), crisp 1px hairline borders, quiet neutral canvases (`#f5f5f7` light / `#0e1113` dark, matching `zapabob/Strata` theme polarity), and a single primary interactive blue (`#0071e3` / `#2997ff`).
- **Hardware-tuned clarity**: CUDA 13.5 `sm_120a` Tensor Core telemetry (BF16 / INT8 / NVFP4 TFLOPS, VRAM slab usage, temperature, power draw) uses tabular monospace numerals (`font-variant-numeric: tabular-nums`) and calm emerald/cyan indicator bars without noisy neon glow.
- **Zero external runtime dependencies**: Single self-contained HTML/CSS/JS bundle embedded into `inkling2-shell` via `include_str!` so it works identically on Windows host loopback (`127.0.0.1:8080`) and offline on the bare-metal Inkling2-OS Linux ISO.

## 2. Layout & Responsive Hierarchy
- **Desktop (>= 900px)**: Two-pane macOS app shell (`272px` frosted sidebar + fluid main content canvas) under a `52px` unified titlebar with segmented navigation.
- **Mobile / Narrow (< 900px)**: Collapsible sidebar drawer toggled from the titlebar; main chat/telemetry view remains 100% first-viewport accessible.

## 3. Accessibility & Security
- **Contrast**: All body text, secondary metadata, and status badges meet WCAG AA (>= 4.5:1 for normal text, >= 3:1 for large text/controls) in both Light and Dark modes.
- **Keyboard & Focus**: Visible `:focus-visible` macOS-style focus rings (`0 0 0 3px rgba(0, 113, 227, 0.35)`), semantic `<button>`, `<nav>`, `<main>`, `aria-live="polite"` for status updates.
- **Loopback-Only Safety**: All API calls target relative same-origin loopback endpoints (`/health`, `/v1/gpu/info`, `/v1/gpu/bench-tc`, `/v1/models`, `/v1/chat/completions`, `/v1/memory/*`, `/v1/lora/*`, `/v1/voice/say`). No external CDN fonts, scripts, or analytics.

## 4. Linux desktop

The native session uses a dark teal canvas (`#17272c`) and slate panels (`#23383f`). The 36 px top bar puts the Inkling2 name on the left and the engine state, network route and clock on the right. The route indicator describes the operating system's default IPv4 route. AI readiness comes from the local engine status endpoint.

The bottom Dock is centered and sized to its launchers. It uses a 16 px corner radius, 32 px system icons and names on hover. Both panels reserve space for app windows. The Dock opens the existing console destinations and the native terminal, Tor Browser and system information.
