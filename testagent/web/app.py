"""FastAPI web application for TestAgent.

Exposes:
  - ``GET  /``         : HTML GUI (embeddable via iframe).
  - ``GET  /health``    : health check.
  - ``GET  /api/config``: current configuration summary.
  - ``POST /api/generate``: generate test cases from requirement text.

The blocking LLM generation runs in a worker thread (via
``run_in_threadpool``) so the async event loop is not blocked.

iframe embedding is enabled by default: ``Content-Security-Policy:
frame-ancestors *`` is sent on every response. Override the allowed
origins with the ``WEB_FRAME_ANCESTORS`` env var (space-separated list).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from testagent.config.models import TestCaseGenInput
from testagent.container import Container
from testagent.generators.testcase_generator import TestCaseGenerator

logger = logging.getLogger(__name__)


def _frame_ancestors() -> str:
    """Return the ``frame-ancestors`` CSP value from env or default ``*``."""
    raw = os.environ.get("WEB_FRAME_ANCESTORS", "*").strip()
    if not raw or raw == "*":
        return "*"
    # Space-separated origins → join with spaces for CSP.
    return " ".join(part.strip() for part in raw.split() if part.strip())


class _FrameEmbedMiddleware(BaseHTTPMiddleware):
    """Allow iframe embedding by setting a permissive ``frame-ancestors``.

    Starlette does not set ``X-Frame-Options`` by default, so we only need
    to emit a CSP ``frame-ancestors`` directive. A reverse proxy can still
    tighten this; this middleware just provides a sane embeddable default.
    """

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = f"frame-ancestors {_frame_ancestors()};"
        # Explicitly drop any inherited framing restriction. Starlette's
        # MutableHeaders has no .pop(), so guard with a membership check.
        if "x-frame-options" in response.headers:
            del response.headers["x-frame-options"]
        return response


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Payload for ``POST /api/generate``."""

    requirements: str = Field(..., description="Requirement document (Markdown/JSON/text).")
    swagger_url: str | None = Field(
        default=None, description="Optional Swagger/OpenAPI URL or path."
    )
    output_format: str = Field(default="json", description="One of: json, csv, markdown.")
    historical_cases: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional historical test cases (list of dicts in the same shape "
            "produced by the JSON exporter). Acts as a baseline; only net-new "
            "cases are generated on top of it."
        ),
    )


class GenerateResponse(BaseModel):
    """Response for ``POST /api/generate``."""

    count: int
    test_cases: list[dict[str, Any]]
    output_format: str
    download_content: str | None = None
    download_filename: str | None = None
    token_usage: str
    historical_count: int = 0


class ConfigResponse(BaseModel):
    """Response for ``GET /api/config``."""

    provider: str
    primary_model: str
    fallback_models: list[str]
    output_language: str
    review_enabled: bool
    review_max_rounds: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(container: Container | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        container: Optional pre-built DI container. A new one is created
            lazily on first request when omitted (deferred so importing the
            web module never requires LLM credentials).
    """
    app = FastAPI(
        title="TestAgent Web",
        description="AI-powered test case generation GUI (iframe-embeddable).",
        version="0.1.0",
    )

    # CORS: permissive to allow embedded cross-origin API calls.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(_FrameEmbedMiddleware)

    # Lazy container holder. We keep a single module-level fallback so the
    # app works even when created without an explicit container.
    _state: dict[str, Container | None] = {"container": container}

    def _get_container() -> Container:
        if _state["container"] is None:
            _state["container"] = Container()
        assert _state["container"] is not None
        return _state["container"]

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/api/config", response_model=ConfigResponse, tags=["meta"])
    def get_config() -> ConfigResponse:
        """Return a sanitized configuration summary for the UI."""
        settings = _get_container().settings
        if settings.azure_llm.enabled:
            provider = "azure"
            primary = settings.azure_llm.deployment
            fallbacks: list[str] = []
        else:
            provider = "openai"
            models = settings.llm.models
            primary = models[0] if models else "(none)"
            fallbacks = models[1:] if len(models) > 1 else []
        return ConfigResponse(
            provider=provider,
            primary_model=primary,
            fallback_models=fallbacks,
            output_language=settings.output_language,
            review_enabled=settings.review_enabled,
            review_max_rounds=settings.review_max_rounds,
        )

    @app.post("/api/generate", response_model=GenerateResponse, tags=["generate"])
    async def generate(req: GenerateRequest) -> GenerateResponse:
        """Generate test cases from requirement text.

        The (blocking) LLM generation is dispatched to a threadpool so the
        event loop stays responsive.
        """
        if not req.requirements.strip():
            raise HTTPException(status_code=400, detail="requirements must not be empty")
        if req.output_format not in ("json", "csv", "markdown"):
            raise HTTPException(
                status_code=400,
                detail="output_format must be one of: json, csv, markdown",
            )

        container = _get_container()

        # Parse historical cases (if any) from raw dicts → TestCase objects.
        historical = (
            [
                tc
                for tc in (TestCaseGenerator._dict_to_testcase(d) for d in req.historical_cases)
                if tc
            ]
            if req.historical_cases
            else []
        )

        try:
            test_cases = await run_in_threadpool(
                _generate_sync,
                container,
                req.requirements,
                req.swagger_url,
                historical,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # surface LLM/config errors to the UI
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        cases_dicts = [TestCaseGenerator._testcase_to_dict(tc) for tc in test_cases]
        download_content, download_filename = _render_output(
            container, test_cases, cases_dicts, req.output_format
        )

        return GenerateResponse(
            count=len(test_cases),
            test_cases=cases_dicts,
            output_format=req.output_format,
            download_content=download_content,
            download_filename=download_filename,
            token_usage=container.llm_client.usage.summary(),
            historical_count=len(historical),
        )

    @app.get("/api/download/{fmt}", tags=["generate"])
    def download_endpoint() -> None:
        """Placeholder kept for documentation; real downloads use the
        ``download_content`` returned by ``/api/generate``."""
        raise HTTPException(status_code=400, detail="Use POST /api/generate instead.")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        """Serve the embedded GUI page."""
        return HTMLResponse(content=_GUI_HTML)

    return app


# ---------------------------------------------------------------------------
# Sync helpers (run in threadpool)
# ---------------------------------------------------------------------------


def _generate_sync(
    container: Container,
    requirements_text: str,
    swagger_url: str | None,
    historical: list[Any],
) -> list[Any]:
    """Run the generation pipeline synchronously.

    The requirement text is written to a temp ``.md`` file so the existing
    :class:`RequirementParser` (which handles Markdown/JSON/text/binary) can
    be reused without duplicating parsing logic.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        req_path = Path(tmpdir) / "requirements.md"
        req_path.write_text(requirements_text, encoding="utf-8")

        req_items = container.requirement_parser.parse(str(req_path))

        endpoints: list[Any] = []
        if swagger_url:
            endpoints = container.swagger_parser.parse(swagger_url)

        if not req_items and not endpoints:
            raise ValueError("No requirements could be parsed from the provided document.")

        return container.testcase_generator.generate(
            TestCaseGenInput(
                endpoints=endpoints,
                requirements=req_items,
                historical_cases=historical,
            )
        )


def _render_output(
    container: Container,
    test_cases: list[Any],
    cases_dicts: list[dict[str, Any]],
    output_format: str,
) -> tuple[str | None, str | None]:
    """Render the test cases into downloadable content for the requested format.

    Returns ``(content, filename)``. For JSON, no separate download blob is
    produced (the array is returned directly in the response); for CSV and
    Markdown a text blob is returned so the browser can offer a download.
    """
    if output_format == "json":
        return None, None
    if output_format == "csv":
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "testcases.csv"
            container.testcase_generator.save_csv(test_cases, out)
            return out.read_text(encoding="utf-8-sig"), "testcases.csv"
    # markdown
    from testagent.config.models import TestCaseReportInput

    report = container.testcase_report.generate(
        TestCaseReportInput(
            test_cases=test_cases,
            output_format="markdown",
            output_language=container.settings.output_language,
        )
    )
    return report, "testcases.md"


# ---------------------------------------------------------------------------
# Embedded GUI (self-contained HTML/CSS/JS, no external assets)
# ---------------------------------------------------------------------------

_GUI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TestAgent - 用例生成</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --panel2: #273449; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --accent2: #0ea5e9; --ok: #22c55e;
    --warn: #f59e0b; --err: #ef4444; --border: #334155; --code: #0b1220;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.55;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
  header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
  header h1 { font-size: 20px; margin: 0; font-weight: 650; }
  header .badge { font-size: 11px; background: var(--panel2); color: var(--accent);
    padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .card h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
  textarea, input, select {
    width: 100%; background: var(--code); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; font-size: 13px; font-family: inherit;
  }
  textarea { min-height: 260px; resize: vertical; font-family: "SFMono-Regular", Menlo, Consolas, monospace; }
  textarea:focus, input:focus, select:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1; min-width: 140px; }
  button {
    background: var(--accent); color: #061018; border: none; border-radius: 8px;
    padding: 11px 18px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background .15s;
  }
  button:hover { background: var(--accent2); }
  button:disabled { opacity: .55; cursor: not-allowed; }
  button.ghost { background: var(--panel2); color: var(--text); border: 1px solid var(--border); }
  button.ghost:hover { background: var(--border); }
  .toolbar { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
  .status { font-size: 12px; color: var(--muted); margin-top: 10px; min-height: 18px; }
  .status.err { color: var(--err); } .status.ok { color: var(--ok); }
  .result { white-space: pre-wrap; word-break: break-word; font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 12px; max-height: 460px; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; }
  th { background: var(--panel2); color: var(--muted); position: sticky; top: 0; }
  tr:nth-child(even) td { background: rgba(255,255,255,.02); }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 10px; background: var(--panel2); border: 1px solid var(--border); }
  .pill.high { color: var(--err); border-color: var(--err); }
  .pill.medium { color: var(--warn); border-color: var(--warn); }
  .pill.low { color: var(--ok); border-color: var(--ok); }
  .meta { font-size: 11px; color: var(--muted); margin-top: 8px; }
  .hidden { display: none; }
  a.dl { color: var(--accent); text-decoration: none; font-size: 12px; }
  a.dl:hover { text-decoration: underline; }
  details { margin-top: 8px; }
  summary { cursor: pointer; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>TestAgent 用例生成</h1>
    <span class="badge" id="modelBadge">model: …</span>
    <span class="badge" id="langBadge">lang: …</span>
  </header>

  <div class="grid">
    <div class="card">
      <h2>输入</h2>
      <label for="req">需求文档（Markdown / JSON / 纯文本）</label>
      <textarea id="req" placeholder="# 用户注册&#10;用户可以使用邮箱注册账号...&#10;验收标准:&#10;- 邮箱格式校验&#10;- 密码不少于 8 位"></textarea>

      <div class="row">
        <div>
          <label for="swagger">Swagger / OpenAPI URL（可选）</label>
          <input id="swagger" type="text" placeholder="https://example.com/openapi.json">
        </div>
        <div>
          <label for="fmt">输出格式</label>
          <select id="fmt">
            <option value="json">JSON（表格预览）</option>
            <option value="csv">CSV（Excel 友好）</option>
            <option value="markdown">Markdown 报告</option>
          </select>
        </div>
      </div>

      <label for="hist">历史用例 JSON（可选，增量基线）</label>
      <textarea id="hist" placeholder='[]' style="min-height:90px"></textarea>

      <div class="toolbar">
        <button id="genBtn" onclick="generate()">生成用例</button>
        <button class="ghost" onclick="loadSample()">载入示例</button>
        <button class="ghost" onclick="clearAll()">清空</button>
      </div>
      <div class="status" id="status"></div>
    </div>

    <div class="card">
      <h2>结果 <span id="countPill" class="pill"></span></h2>
      <div id="downloadBar" class="hidden" style="margin-bottom:10px"></div>
      <div id="resultView" class="result">
        <span style="color:var(--muted)">生成结果将显示在这里。</span>
      </div>
      <div class="meta" id="meta"></div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);

async function init() {
  try {
    const r = await fetch('/api/config');
    const c = await r.json();
    $('modelBadge').textContent = 'model: ' + (c.primary_model || '—') +
      (c.fallback_models && c.fallback_models.length ? ' +' + c.fallback_models.length : '');
    $('langBadge').textContent = 'lang: ' + c.output_language;
  } catch (e) { /* ignore */ }
}

function setStatus(msg, kind) {
  const s = $('status');
  s.textContent = msg || '';
  s.className = 'status' + (kind ? ' ' + kind : '');
}

function loadSample() {
  $('req').value = '# 用户注册\\n用户可以使用邮箱注册账号，注册成功后收到验证邮件。\\n\\n验收标准:\\n- 邮箱必须为合法格式\\n- 密码不少于 8 位且含数字与字母\\n- 重复邮箱注册返回 409\\n';
}

function clearAll() {
  $('req').value = ''; $('swagger').value = ''; $('hist').value = '';
  $('resultView').innerHTML = '<span style="color:var(--muted)">已清空。</span>';
  $('countPill').textContent = ''; $('downloadBar').classList.add('hidden');
  $('meta').textContent = ''; setStatus('', '');
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (ch) => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[ch]));
}

function arrToList(arr) {
  if (!arr || !arr.length) return '<span style="color:var(--muted)">—</span>';
  return '<ul style="margin:0;padding-left:16px">' +
    arr.map((x) => '<li>' + escapeHtml(x) + '</li>').join('') + '</ul>';
}

function priorityClass(p) { return p || ''; }

function renderTable(cases) {
  if (!cases.length) return '<span style="color:var(--muted)">无用例。</span>';
  let html = '<table><thead><tr>' +
    ['ID','标题','端点','类型','优先级','前置条件','步骤','预期结果','标签'].map((h) => '<th>'+h+'</th>').join('') +
    '</tr></thead><tbody>';
  for (const c of cases) {
    html += '<tr>' +
      '<td>' + escapeHtml(c.id) + '</td>' +
      '<td>' + escapeHtml(c.title) + '</td>' +
      '<td>' + escapeHtml(c.endpoint) + '</td>' +
      '<td>' + escapeHtml(c.test_type) + '</td>' +
      '<td><span class="pill ' + priorityClass(c.priority) + '">' + escapeHtml(c.priority) + '</span></td>' +
      '<td>' + arrToList(c.preconditions) + '</td>' +
      '<td>' + arrToList(c.steps) + '</td>' +
      '<td>' + arrToList(c.expected_results) + '</td>' +
      '<td>' + (c.tags && c.tags.length ? c.tags.map((t)=>'<span class="pill">'+escapeHtml(t)+'</span>').join(' ') : '—') + '</td>' +
      '</tr>';
  }
  html += '</tbody></table>';
  return html;
}

function renderRaw(content) {
  return '<pre>' + escapeHtml(content) + '</pre>';
}

function triggerDownload(filename, content, fmt) {
  const mime = fmt === 'csv' ? 'text/csv;charset=utf-8' : 'text/markdown;charset=utf-8';
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

async function generate() {
  const req = $('req').value.trim();
  if (!req) { setStatus('请输入需求文档。', 'err'); return; }
  const swagger = $('swagger').value.trim() || null;
  const fmt = $('fmt').value;
  let hist = null;
  const histText = $('hist').value.trim();
  if (histText) {
    try { hist = JSON.parse(histText); if (!Array.isArray(hist)) throw 0; }
    catch (e) { setStatus('历史用例 JSON 格式错误，需为数组。', 'err'); return; }
  }

  const btn = $('genBtn'); btn.disabled = true;
  setStatus('正在生成（可能需要数十秒）…');
  $('resultView').innerHTML = '<span style="color:var(--muted)">生成中…</span>';
  $('countPill').textContent = '';
  $('downloadBar').classList.add('hidden');

  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ requirements: req, swagger_url: swagger, output_format: fmt, historical_cases: hist })
    });
    const data = await r.json();
    if (!r.ok) {
      setStatus('失败：' + (data.detail || r.status), 'err');
      $('resultView').innerHTML = '<span style="color:var(--err)">' + escapeHtml(JSON.stringify(data)) + '</span>';
      return;
    }
    $('countPill').textContent = data.count + ' 条' + (data.historical_count ? ' (基线 ' + data.historical_count + ')' : '');
    if (fmt === 'json') {
      $('resultView').innerHTML = renderTable(data.test_cases);
    } else if (data.download_content) {
      $('resultView').innerHTML = renderRaw(data.download_content);
    } else {
      $('resultView').innerHTML = renderRaw(JSON.stringify(data.test_cases, null, 2));
    }
    if (data.download_content && data.download_filename) {
      const bar = $('downloadBar');
      bar.classList.remove('hidden');
      bar.innerHTML = '<a class="dl" href="#" onclick="triggerDownload(' +
        JSON.stringify(data.download_filename) + ',' +
        JSON.stringify(data.download_content) + ',' + JSON.stringify(fmt) +
        ');return false">⬇ 下载 ' + escapeHtml(data.download_filename) + '</a>';
    }
    setStatus('完成。' + (data.token_usage ? ' Token: ' + data.token_usage : ''), 'ok');
    $('meta').textContent = 'format=' + data.output_format + ' | count=' + data.count;
  } catch (e) {
    setStatus('网络错误：' + e, 'err');
  } finally {
    btn.disabled = false;
  }
}

init();
</script>
</body>
</html>"""
