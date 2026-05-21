"""
Estilos visuales para la interfaz Streamlit.
"""

from __future__ import annotations

import streamlit as st


def apply_custom_styles() -> None:
    """
    Inyecta CSS personalizado para dar una apariencia mas pulida a la app.
    """
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f9fd;
            --panel: #ffffff;
            --panel-border: #e4e9f2;
            --ink: #091635;
            --muted: #69738a;
            --accent: #ff5a4f;
            --accent-soft: #fff1ef;
            --success: #22c55e;
            --success-soft: #ecfdf3;
            --info: #2f80ed;
            --info-soft: #edf6ff;
            --warning-soft: #fff7e8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 90, 79, 0.08), transparent 28rem),
                linear-gradient(180deg, #fbfcff 0%, var(--app-bg) 100%);
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1240px;
            padding: 1.25rem 2rem 2.5rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .app-hero {
            align-items: center;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1rem;
        }

        .app-brand {
            align-items: center;
            display: flex;
            gap: 1rem;
            min-width: 0;
        }

        .logo-mark {
            align-items: center;
            background: linear-gradient(135deg, #ffc27d 0%, #ff8b5b 58%, #ff6f61 100%);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            box-shadow: 0 14px 30px rgba(255, 107, 88, 0.22);
            color: #ffffff;
            display: flex;
            flex: 0 0 auto;
            font-size: 1.8rem;
            font-weight: 800;
            height: 56px;
            justify-content: center;
            width: 56px;
        }

        .logo-image {
            display: block;
            flex: 0 0 auto;
            height: 56px;
            width: 56px;
        }

        .app-title {
            color: var(--ink);
            font-size: clamp(1.75rem, 3vw, 2.75rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.05;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0.35rem 0 0;
        }

        .header-actions {
            display: flex;
            gap: 0.5rem;
        }

        .header-pill {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            color: #40506f;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 700;
            gap: 0.35rem;
            min-height: 38px;
            padding: 0.45rem 0.75rem;
            white-space: nowrap;
        }

        .status-banner {
            align-items: center;
            border: 1px solid;
            border-radius: 8px;
            display: flex;
            font-weight: 650;
            gap: 0.75rem;
            margin: 0.25rem 0 1.1rem;
            min-height: 46px;
            padding: 0.85rem 1rem;
        }

        .status-banner.success {
            background: var(--success-soft);
            border-color: #bbf7d0;
            color: #166534;
        }

        .status-banner.error {
            background: #fff1f2;
            border-color: #fecdd3;
            color: #9f1239;
        }

        .status-icon {
            align-items: center;
            border-radius: 999px;
            color: #ffffff;
            display: inline-flex;
            font-size: 0.8rem;
            height: 24px;
            justify-content: center;
            width: 24px;
        }

        .status-banner.success .status-icon {
            background: var(--success);
        }

        .status-banner.error .status-icon {
            background: #f43f5e;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--panel-border);
            gap: 1.5rem;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            color: #536079;
            font-size: 0.95rem;
            font-weight: 700;
            height: 54px;
            letter-spacing: 0;
            padding: 0 0.25rem;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
        }

        div[data-testid="stTabs"] button[aria-selected="true"]::after {
            background: var(--accent);
            border-radius: 999px 999px 0 0;
            bottom: -1px;
            content: "";
            height: 3px;
            left: 0;
            position: absolute;
            right: 0;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        .panel-title {
            align-items: center;
            color: var(--ink);
            display: flex;
            font-size: 1.35rem;
            font-weight: 800;
            gap: 0.55rem;
            letter-spacing: 0;
            margin: 0.1rem 0 1rem;
        }

        .panel-title.small {
            font-size: 1.05rem;
            margin-bottom: 0.85rem;
        }

        .panel-kicker {
            color: #344054;
            font-size: 0.86rem;
            font-weight: 700;
            margin: 0.6rem 0 0.45rem;
        }

        div[role="radiogroup"] {
            gap: 0.45rem;
        }

        div[role="radiogroup"] label {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background: var(--accent-soft);
            border-color: #ffb3aa;
            color: var(--accent);
        }

        [data-testid="stFileUploaderDropzone"] {
            background:
                linear-gradient(180deg, rgba(255, 246, 244, 0.7), rgba(255, 255, 255, 0.95));
            border: 1.5px dashed #ff9c92;
            border-radius: 8px;
            min-height: 150px;
        }

        [data-testid="stFileUploaderDropzone"] svg {
            color: var(--accent);
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
            font-weight: 750;
            min-height: 40px;
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        .preview-placeholder {
            align-items: center;
            aspect-ratio: 1 / 1;
            background:
                linear-gradient(135deg, #f4f6fa 0%, #ffffff 100%);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            color: #a7b0c2;
            display: flex;
            flex-direction: column;
            font-weight: 700;
            justify-content: center;
            min-height: 165px;
            text-align: center;
        }

        .preview-icon {
            font-size: 3.5rem;
            line-height: 1;
            margin-bottom: 0.4rem;
        }

        .preview-caption {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.45rem;
            text-align: center;
        }

        .result-stack {
            display: grid;
            gap: 0.55rem;
        }

        .result-stat {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            min-height: 74px;
            padding: 0.75rem 0.85rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .result-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .result-badge {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            flex: 0 0 auto;
            font-size: 1.1rem;
            height: 44px;
            justify-content: center;
            width: 44px;
        }

        .badge-red {
            background: #ffe4e1;
            color: var(--accent);
        }

        .badge-blue {
            background: #eaf3ff;
            color: var(--info);
        }

        .badge-orange {
            background: var(--warning-soft);
            color: #f59e0b;
        }

        .badge-green {
            background: #dcfce7;
            color: #15803d;
        }

        .badge-yellow {
            background: #fef3c7;
            color: #d97706;
        }

        .confidence-level-badge {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 800;
            min-height: 28px;
            padding: 0.35rem 0.75rem;
            white-space: nowrap;
        }

        .confidence-high {
            background: #dcfce7;
            color: #15803d;
        }

        .confidence-medium {
            background: #fef3c7;
            color: #d97706;
        }

        .confidence-low {
            background: #fee2e2;
            color: #dc2626;
        }

        .confidence-total,
        .confidence-average {
            background: #eaf3ff;
            color: #2563eb;
        }

        .confidence-metric-card {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            min-height: 74px;
            padding: 0.75rem 0.85rem;
        }

        .confidence-metric-card.confidence-high {
            border-color: #bbf7d0;
        }

        .confidence-metric-card.confidence-medium {
            border-color: #fde68a;
        }

        .confidence-metric-card.confidence-low {
            border-color: #fecaca;
        }

        .confidence-metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .confidence-metric-value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .confidence-metric-icon {
            border-radius: 999px;
            flex: 0 0 auto;
            height: 14px;
            width: 14px;
        }

        .confidence-metric-card.confidence-total .confidence-metric-icon,
        .confidence-metric-card.confidence-average .confidence-metric-icon {
            background: #2563eb;
        }

        .confidence-metric-card.confidence-high .confidence-metric-icon {
            background: #16a34a;
        }

        .confidence-metric-card.confidence-medium .confidence-metric-icon {
            background: #f59e0b;
        }

        .confidence-metric-card.confidence-low .confidence-metric-icon {
            background: #dc2626;
        }

        .info-strip {
            align-items: center;
            background: linear-gradient(90deg, #eff7ff 0%, #f7fbff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            color: #1f3b64;
            display: flex;
            gap: 0.85rem;
            margin-top: 1rem;
            min-height: 64px;
            padding: 0.9rem 1rem;
        }

        .info-icon {
            align-items: center;
            background: var(--info);
            border-radius: 999px;
            color: #ffffff;
            display: inline-flex;
            flex: 0 0 auto;
            font-weight: 800;
            height: 28px;
            justify-content: center;
            width: 28px;
        }

        .info-title {
            color: var(--ink);
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .info-copy {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .confirm-grid {
            border-top: 1px solid var(--panel-border);
            margin-top: 1rem;
            padding-top: 1rem;
        }

        .batch-file-list {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin-top: 0.75rem;
            overflow: hidden;
        }

        .batch-file-row {
            align-items: center;
            background: #ffffff;
            border-bottom: 1px solid var(--panel-border);
            display: grid;
            gap: 0.75rem;
            grid-template-columns: minmax(0, 1fr) 96px 112px 24px;
            min-height: 54px;
            padding: 0.45rem 0.75rem;
        }

        .batch-file-row:last-child {
            border-bottom: 0;
        }

        .batch-file-main {
            align-items: center;
            display: flex;
            gap: 0.65rem;
            min-width: 0;
        }

        .batch-file-thumb,
        .batch-file-icon {
            border-radius: 8px;
            flex: 0 0 auto;
            height: 34px;
            width: 34px;
        }

        .batch-file-thumb {
            border: 1px solid #edf1f7;
            object-fit: cover;
        }

        .batch-file-icon {
            align-items: center;
            background: var(--accent-soft);
            color: var(--accent);
            display: inline-flex;
            font-weight: 800;
            justify-content: center;
        }

        .batch-file-name {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 750;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .batch-file-size {
            color: #526079;
            font-size: 0.84rem;
            font-weight: 700;
            text-align: right;
        }

        .batch-status {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 800;
            justify-content: center;
            min-height: 28px;
            padding: 0 0.7rem;
        }

        .batch-status.ready,
        .batch-status.uploaded {
            background: #eaf3ff;
            color: #2563eb;
        }

        .batch-status.processed {
            background: #ecfdf3;
            color: #15803d;
        }

        .batch-status.pending {
            background: #fff7e8;
            color: #d97706;
        }

        .batch-status.error {
            background: #fff1f2;
            color: #e11d48;
        }

        .batch-remove {
            color: #7890ad;
            font-size: 1.2rem;
            line-height: 1;
            text-align: center;
        }

        .batch-action-row {
            display: flex;
            gap: 0.75rem;
            margin-top: 0.8rem;
        }

        .batch-summary {
            display: grid;
            gap: 0.55rem;
        }

        .batch-summary-row {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            min-height: 48px;
            padding: 0.65rem 0.75rem;
        }

        .batch-summary-label {
            align-items: center;
            color: #33405c;
            display: flex;
            font-size: 0.86rem;
            font-weight: 750;
            gap: 0.55rem;
        }

        .batch-summary-icon {
            align-items: center;
            border-radius: 8px;
            display: inline-flex;
            font-size: 0.85rem;
            height: 28px;
            justify-content: center;
            width: 28px;
        }

        .batch-summary-icon.blue {
            background: #eaf3ff;
            color: #2f80ed;
        }

        .batch-summary-icon.green {
            background: #ecfdf3;
            color: #16a34a;
        }

        .batch-summary-icon.orange {
            background: #fff1e7;
            color: #f97316;
        }

        .batch-summary-icon.purple {
            background: #f5edff;
            color: #8b5cf6;
        }

        .batch-state.ok {
            color: #16a34a;
        }

        .batch-state.muted {
            color: #64748b;
        }

        .batch-progress-block,
        .batch-category-block {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        .batch-progress-head {
            align-items: center;
            color: var(--ink);
            display: flex;
            font-size: 0.86rem;
            font-weight: 800;
            justify-content: space-between;
            margin-bottom: 0.55rem;
        }

        .batch-progress-track {
            background: #e8edf5;
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
        }

        .batch-progress-track span {
            background: linear-gradient(90deg, var(--accent), #ff9a74);
            border-radius: inherit;
            display: block;
            height: 100%;
        }

        .batch-progress-caption {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        .batch-category-title {
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .batch-category-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .category-chip {
            background: #fff1ef;
            border-radius: 999px;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 800;
            min-height: 28px;
            padding: 0.35rem 0.75rem;
        }

        .category-chip.muted {
            background: #f3f6fb;
            color: #526079;
        }

        .stDataFrame,
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .app-hero {
                align-items: flex-start;
                flex-direction: column;
            }

            .header-actions {
                flex-wrap: wrap;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0.75rem;
            }

            .batch-file-row {
                grid-template-columns: minmax(0, 1fr);
            }

            .batch-file-size,
            .batch-remove {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
