#!/usr/bin/env python3
"""
prompt-forge: Transform raw, sloppy prompts into production-quality prompts.
Supports Anthropic, OpenAI, Gemini, Groq, Ollama, MLX, and any
OpenAI-compatible local/remote endpoint.

Usage:
    python prompt_forge.py "your raw prompt here"
    python prompt_forge.py --provider openai "summarize this doc"
    python prompt_forge.py --provider groq --model llama-3.3-70b-versatile "find bugs"
    python prompt_forge.py --provider ollama --model llama3.2 "analyze logs"
    python prompt_forge.py --provider mlx --model mlx-community/Llama-3.2-3B "explain this"
    python prompt_forge.py --provider custom --base-url http://localhost:1234/v1 --model mymodel "..."
    python prompt_forge.py --list-providers
"""

from __future__ import annotations

import sys
import os
import re
import json
import argparse
import datetime
from pathlib import Path
from typing import Iterator

# Load .env from the same directory as this script (silently ignored if absent)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to environment variables

# ── Provider registry ─────────────────────────────────────────────────────────
# sdk: "anthropic" | "openai" | "gemini"
# env_key: environment variable name for the API key (None = no key needed)
# default_model: model used when --model is not specified
# base_url: API base URL override (None = SDK default)
# local: True means it's a local endpoint (no key check, reachability warning)

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "sdk": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4-6",
        "base_url": None,
        "local": False,
    },
    "openai": {
        "sdk": "openai",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "base_url": None,
        "local": False,
    },
    "gemini": {
        "sdk": "gemini",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-3-flash-preview",
        "base_url": None,
        "local": False,
    },
    "groq": {
        "sdk": "openai",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "local": False,
    },
    "ollama": {
        "sdk": "openai",
        "env_key": None,
        "default_model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "local": True,
    },
    "mlx": {
        "sdk": "openai",
        "env_key": None,
        "default_model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "base_url": "http://localhost:8080/v1",
        "local": True,
    },
    # "custom" is handled separately via --base-url
}

VERSIONS_FILE = Path(__file__).parent / "prompt_versions.json"

# ── Meta-prompt ───────────────────────────────────────────────────────────────

META_PROMPT_TEMPLATE = """You are a prompt engineering expert. Transform the raw prompt below into a production-quality prompt.

Here is a concrete example of what this transformation looks like:

<transformation_example>
<raw>analyze this dataset and find problems</raw>
<rewritten>
<system>
You are a senior data quality analyst with expertise in statistical analysis, data profiling, and root-cause investigation. You produce precise, actionable audit reports.
</system>

<output_format>
Respond with a structured markdown report:
1. **Executive Summary** — 2–3 sentence overview of findings
2. **Issues Found** — table with columns: Issue | Severity (Critical/High/Medium/Low) | Affected Column(s) | Recommended Fix
3. **Suggested Next Steps** — numbered list of prioritized actions
</output_format>

<instructions>
Analyze the provided dataset for: missing/null values, outliers, type mismatches, duplicates, and referential integrity violations.
</instructions>

<examples>
<example>
  <input>id,name,age,email\n1,Alice,29,alice@example.com\n2,,null,bob@\n3,Alice,29,alice@example.com</input>
  <output>
**Executive Summary**: The dataset (3 rows) has 2 quality issues: one incomplete record and one exact duplicate. Remediation required before downstream use.

| Issue | Severity | Affected Column(s) | Recommended Fix |
|---|---|---|---|
| Missing name, invalid age | Critical | name, age | Backfill from source or flag for manual review |
| Malformed email | High | email | Validate against RFC 5322 |
| Exact duplicate row | Medium | all | Deduplicate on (name, age, email) |

**Suggested Next Steps**:
1. Deduplicate before any joins
2. Add NOT NULL constraints on name and age
3. Add email format validation at ingestion
  </output>
</example>
<example>
  <input>sales.csv with 10k rows — manager says revenue looks wrong</input>
  <output>
**Executive Summary**: Revenue column shows 340 negative values and a 6× spike on 2024-03-15 inconsistent with surrounding days. Likely a sign-reversal bug in ETL and a batch reprocessing artifact.

| Issue | Severity | Affected Column(s) | Recommended Fix |
|---|---|---|---|
| Negative revenue values | Critical | revenue | Audit ETL sign convention; abs() as interim fix |
| Anomalous spike 2024-03-15 | High | revenue, date | Investigate batch job logs for that date |

**Suggested Next Steps**:
1. Pull ETL logs for 2024-03-15
2. Compare raw source vs loaded values for negative rows
3. Add a revenue > 0 assertion to the pipeline
  </output>
</example>
</examples>

<context>
{{DATASET_OR_SCHEMA}}
</context>
</rewritten>
</transformation_example>

Your rewritten prompt MUST include all of the following:

1. **System Role**: A `<system>` block defining the AI's role, expertise, and behavior.
2. **Structured Output Format**: An `<output_format>` block specifying exactly how the response should be structured.
3. **XML Delimiters**: Use XML tags to delimit sections (`<context>`, `<input>`, `<instructions>`, `<constraints>` as appropriate).
4. **Few-Shot Examples**: An `<examples>` block with 2–3 concrete, realistic examples drawn from the domain — not placeholder variables like {{EXAMPLE_INPUT_1}}. Infer plausible inputs and outputs from the raw prompt's intent.
{domain_instruction}

Raw prompt to transform:
<raw_prompt>
{raw_prompt}
</raw_prompt>

Return ONLY the rewritten production-quality prompt. Do not include any explanation or commentary."""


def build_meta_prompt(raw_prompt: str, domain: str | None = None) -> str:
    domain_instruction = ""
    if domain:
        domain_instruction = (
            f"\n5. **Domain Context**: Tailor the rewritten prompt specifically for "
            f"the **{domain}** domain, using relevant terminology and conventions."
        )
    return META_PROMPT_TEMPLATE.format(
        raw_prompt=raw_prompt,
        domain_instruction=domain_instruction,
    )


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _stream_anthropic(system: str, user: str, model: str, max_tokens: int) -> Iterator[str]:
    """Stream via the Anthropic SDK."""
    try:
        import anthropic as _anthropic
    except ImportError:
        _die("anthropic SDK not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _die("ANTHROPIC_API_KEY environment variable is not set.")

    client = _anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        yield from stream.text_stream


def _stream_openai_compat(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    base_url: str | None,
    api_key: str | None,
) -> Iterator[str]:
    """Stream via the OpenAI SDK (works for OpenAI, Groq, Ollama, MLX, custom)."""
    try:
        from openai import OpenAI
    except ImportError:
        _die("openai SDK not installed. Run: pip install openai")

    # Local providers (Ollama, MLX) don't require a real key
    effective_key = api_key or "local"

    client = OpenAI(api_key=effective_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_gemini(system: str, user: str, model: str, max_tokens: int) -> Iterator[str]:
    """Stream via the google-genai SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _die(
            "google-genai SDK not installed. Run: pip install google-genai\n"
            "Or use the OpenAI-compatible endpoint with:\n"
            "  --provider custom "
            "--base-url https://generativelanguage.googleapis.com/v1beta/openai/ "
            "--api-key $GEMINI_API_KEY"
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        _die("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set.")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
    )
    # SDK accepts bare names like "gemini-3-flash-preview"; strip prefix if passed
    clean_model = model.removeprefix("models/")
    for chunk in client.models.generate_content_stream(
        model=clean_model,
        contents=user,
        config=config,
    ):
        # Gemini 3 thinking models include thought_signature parts — skip them
        try:
            if chunk.text:
                yield chunk.text
        except Exception:
            pass  # non-text chunk (thought signature, etc.) — skip


def stream_completion(
    system: str,
    user: str,
    provider_cfg: dict,
    model: str,
    max_tokens: int,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Iterator[str]:
    """Dispatch to the correct streaming backend."""
    sdk = provider_cfg["sdk"]
    effective_base_url = base_url or provider_cfg.get("base_url")

    if sdk == "anthropic":
        yield from _stream_anthropic(system, user, model, max_tokens)
    elif sdk == "openai":
        effective_key = api_key or (
            os.environ.get(provider_cfg["env_key"]) if provider_cfg.get("env_key") else None
        )
        yield from _stream_openai_compat(
            system, user, model, max_tokens, effective_base_url, effective_key
        )
    elif sdk == "gemini":
        yield from _stream_gemini(system, user, model, max_tokens)
    else:
        _die(f"Unknown SDK type: {sdk!r}")


# ── Core logic ────────────────────────────────────────────────────────────────

def rewrite_prompt(
    raw_prompt: str,
    provider_cfg: dict,
    model: str,
    domain: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    meta = build_meta_prompt(raw_prompt, domain)
    system = "You are a prompt engineering expert. Follow instructions exactly."

    print("⟳ Rewriting prompt...\n", file=sys.stderr)
    parts: list[str] = []
    try:
        for chunk in stream_completion(
            system=system,
            user=meta,
            provider_cfg=provider_cfg,
            model=model,
            max_tokens=4096,
            api_key=api_key,
            base_url=base_url,
        ):
            print(chunk, end="", flush=True)
            parts.append(chunk)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _die(f"API error ({provider_cfg.get('sdk', 'unknown')}): {e}")

    print()  # newline after stream
    return "".join(parts)


def run_prompt(
    prompt: str,
    provider_cfg: dict,
    model: str,
    sample_input: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Execute a prompt (raw or rewritten) against the selected provider."""
    user_message = sample_input or "Please demonstrate your capabilities with a brief example."

    # Extract <system> or <s> block if present
    system_content = "You are a helpful AI assistant."
    system_match = re.search(r'<(system|s)>(.*?)</\1>', prompt, re.DOTALL | re.IGNORECASE)
    if system_match:
        system_content = system_match.group(2).strip()
        before = prompt[:system_match.start()].strip()
        after = prompt[system_match.end():].strip()
        user_content = "\n\n".join(filter(None, [before, after, user_message]))
    else:
        user_content = f"{prompt}\n\n{user_message}"

    parts: list[str] = []
    try:
        for chunk in stream_completion(
            system=system_content,
            user=user_content,
            provider_cfg=provider_cfg,
            model=model,
            max_tokens=2048,
            api_key=api_key,
            base_url=base_url,
        ):
            print(chunk, end="", flush=True)
            parts.append(chunk)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _die(f"API error ({provider_cfg.get('sdk', 'unknown')}): {e}")

    print()
    return "".join(parts)


def _structural_score(text: str) -> dict:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "headers": bool(re.search(r'^#{1,3} |\*\*[A-Z]', text, re.MULTILINE)),
        "table": bool(re.search(r'\|.+\|\n\|[-| :]+\|', text)),
        "numbered_list": bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE)),
        "bullet_list": bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE)),
    }


def compare_prompts(
    raw_prompt: str,
    rewritten_prompt: str,
    sample_input: str,
    provider_cfg: dict,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    sep = "─" * 70
    kwargs = dict(provider_cfg=provider_cfg, model=model, api_key=api_key, base_url=base_url)

    print(f"\n{sep}\nORIGINAL PROMPT OUTPUT\n{sep}")
    orig = run_prompt(raw_prompt, sample_input=sample_input, **kwargs)

    print(f"\n{sep}\nREWRITTEN PROMPT OUTPUT\n{sep}")
    rewr = run_prompt(rewritten_prompt, sample_input=sample_input, **kwargs)

    o, r = _structural_score(orig), _structural_score(rewr)
    print(f"\n{sep}\nCOMPARISON SUMMARY\n{sep}")
    print(f"  {'Metric':<20}  {'Original':>10}  {'Rewritten':>10}")
    print("  " + "─" * 44)
    print(f"  {'Characters':<20}  {o['chars']:>10,}  {r['chars']:>10,}")
    print(f"  {'Words':<20}  {o['words']:>10,}  {r['words']:>10,}")
    for key, label in [("headers", "Headers"), ("table", "Table"), ("numbered_list", "Numbered list"), ("bullet_list", "Bullet list")]:
        print(f"  {label:<20}  {'✓' if o[key] else '✗':>10}  {'✓' if r[key] else '✗':>10}")


# ── Version management ────────────────────────────────────────────────────────

def _load_versions() -> list[dict]:
    if VERSIONS_FILE.exists():
        try:
            return json.loads(VERSIONS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return []


def save_version(
    name: str,
    raw_prompt: str,
    rewritten_prompt: str,
    domain: str | None,
    provider: str,
    model: str,
) -> None:
    versions = _load_versions()
    entry = {
        "id": len(versions) + 1,
        "name": name,
        "domain": domain,
        "provider": provider,
        "model": model,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "raw_prompt": raw_prompt,
        "rewritten_prompt": rewritten_prompt,
    }
    versions.append(entry)
    VERSIONS_FILE.write_text(json.dumps(versions, indent=2))
    print(f"\n✓ Saved version #{entry['id']}: '{name}' → {VERSIONS_FILE}", file=sys.stderr)


def list_versions() -> None:
    versions = _load_versions()
    if not versions:
        print("No saved versions found.", file=sys.stderr)
        return
    print(f"\n{'ID':>4}  {'Name':<25}  {'Provider':<12}  {'Domain':<18}  {'Created'}")
    print("─" * 82)
    for v in versions:
        domain = v.get("domain") or "—"
        provider = v.get("provider", "—")
        created = v["created_at"][:19].replace("T", " ")
        print(f"{v['id']:>4}  {v['name']:<25}  {provider:<12}  {domain:<18}  {created}")


def show_version(version_id: int) -> None:
    versions = _load_versions()
    v = next((v for v in versions if v["id"] == version_id), None)
    if not v:
        _die(f"Version #{version_id} not found.")
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Version #{v['id']}: {v['name']}")
    print(f"Provider : {v.get('provider', '—')}  |  Model: {v.get('model', '—')}")
    print(f"Domain   : {v.get('domain') or '—'}")
    print(f"Created  : {v['created_at'][:19].replace('T', ' ')}")
    print(f"{sep}\n── RAW PROMPT ──\n")
    print(v["raw_prompt"])
    print(f"\n── REWRITTEN PROMPT ──\n")
    print(v["rewritten_prompt"])


# ── Provider listing ──────────────────────────────────────────────────────────

def list_providers() -> None:
    print("\nAvailable providers:\n")
    print(f"  {'Name':<12}  {'Default model':<45}  {'API key env var':<22}  Status")
    print("  " + "─" * 100)
    for name, cfg in PROVIDERS.items():
        model = cfg["default_model"]
        env_key = cfg.get("env_key") or "—"
        if cfg.get("local"):
            status = "local (no key needed)"
        elif cfg.get("env_key") and os.environ.get(cfg["env_key"]):
            status = "✓ key set"
        else:
            status = "✗ key not set"
        print(f"  {name:<12}  {model:<45}  {env_key:<22}  {status}")
    print(
        "\n  custom      (any OpenAI-compatible endpoint — use --base-url and --model)\n"
    )
    print("Tip: set API keys as environment variables, e.g.")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export GEMINI_API_KEY=AI...")
    print("  export GROQ_API_KEY=gsk_...")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _resolve_provider(name: str, base_url: str | None) -> tuple[str, dict]:
    """Return (canonical_name, provider_cfg). Handles 'custom' provider."""
    if name == "custom":
        if not base_url:
            _die("--provider custom requires --base-url <URL>")
        cfg = {
            "sdk": "openai",
            "env_key": None,
            "default_model": "default",
            "base_url": base_url,
            "local": True,
        }
        return "custom", cfg
    if name not in PROVIDERS:
        _die(
            f"Unknown provider '{name}'. "
            f"Valid options: {', '.join(list(PROVIDERS) + ['custom'])}"
        )
    return name, PROVIDERS[name]


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-forge",
        description="Transform raw prompts into production-quality prompts using any LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Providers:
  anthropic   Claude models via Anthropic API  (ANTHROPIC_API_KEY)
  openai      GPT models via OpenAI API        (OPENAI_API_KEY)
  gemini      Gemini models via Google AI      (GEMINI_API_KEY)
  groq        Fast inference via Groq          (GROQ_API_KEY)
  ollama      Local models via Ollama          (no key needed)
  mlx         Local models via MLX on Mac      (no key needed)
  custom      Any OpenAI-compatible endpoint   (use --base-url)

Examples:
  # Default (Anthropic claude-opus-4-6)
  python prompt_forge.py "look at this data and find problems"

  # OpenAI GPT-4o
  python prompt_forge.py --provider openai "find bugs in this code"

  # Gemini
  python prompt_forge.py --provider gemini --model gemini-2.0-flash "summarize this"

  # Groq (fast Llama inference)
  python prompt_forge.py --provider groq --model llama-3.3-70b-versatile "classify this"

  # Ollama local model
  python prompt_forge.py --provider ollama --model llama3.2 "analyze these logs"

  # MLX local model on Mac
  python prompt_forge.py --provider mlx --model mlx-community/Llama-3.2-3B-Instruct-4bit "explain"

  # Any OpenAI-compatible endpoint (e.g. LM Studio, vLLM, llamafile)
  python prompt_forge.py --provider custom --base-url http://localhost:1234/v1 --model phi-4 "help"

  # Domain-aware rewrite
  python prompt_forge.py --domain "data engineering" "find issues in the pipeline"

  # Compare original vs rewritten on a sample input
  python prompt_forge.py --compare --sample-input "col1,col2\\n1,null" "find data problems"

  # Save a named version
  python prompt_forge.py --save "data-check-v1" "look at this data"

  # Version management
  python prompt_forge.py --list-versions
  python prompt_forge.py --show-version 3

  # List providers and key status
  python prompt_forge.py --list-providers
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The raw prompt to rewrite. Reads from stdin if not provided.",
    )
    parser.add_argument(
        "--provider", "-p",
        metavar="NAME",
        default="anthropic",
        help="LLM provider to use (default: anthropic). See --list-providers.",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="MODEL",
        default=None,
        help="Model name override. Defaults to the provider's default model.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        default=None,
        help="Base URL for OpenAI-compatible endpoints (used with --provider custom or to override).",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="API key override (default: read from environment variable).",
    )
    parser.add_argument(
        "--domain", "-d",
        metavar="DOMAIN",
        help="Domain context for the rewrite (e.g., 'data engineering', 'support triage').",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Run both original and rewritten prompts against a sample input and compare.",
    )
    parser.add_argument(
        "--sample-input", "-i",
        metavar="TEXT",
        help="Sample input to use with --compare.",
    )
    parser.add_argument(
        "--save", "-s",
        metavar="NAME",
        help="Save the rewritten prompt with a name to prompt_versions.json.",
    )
    parser.add_argument(
        "--list-versions", "-l",
        action="store_true",
        help="List all saved prompt versions.",
    )
    parser.add_argument(
        "--show-version",
        metavar="ID",
        type=int,
        help="Show a saved prompt version by ID.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all available providers and their API key status.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Informational subcommands ─────────────────────────────────────────────
    if args.list_providers:
        list_providers()
        return

    if args.list_versions:
        list_versions()
        return

    if args.show_version is not None:
        show_version(args.show_version)
        return

    # ── Resolve provider ──────────────────────────────────────────────────────
    provider_name, provider_cfg = _resolve_provider(args.provider, args.base_url)
    model = args.model or provider_cfg["default_model"]

    # Warn if no model specified for custom provider
    if provider_name == "custom" and not args.model:
        print(
            "Warning: --provider custom with no --model; using 'default'. "
            "Specify --model <name> for your endpoint.",
            file=sys.stderr,
        )

    # ── Read raw prompt ───────────────────────────────────────────────────────
    if args.prompt:
        raw_prompt = args.prompt.strip()
    elif not sys.stdin.isatty():
        raw_prompt = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(0)

    if not raw_prompt:
        _die("Prompt cannot be empty.")

    if args.compare and not args.sample_input:
        _die("--compare requires --sample-input TEXT.")

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "─" * 70
    print(f"\n{sep}", file=sys.stderr)
    print("prompt-forge", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"Provider : {provider_name}", file=sys.stderr)
    print(f"Model    : {model}", file=sys.stderr)
    if args.domain:
        print(f"Domain   : {args.domain}", file=sys.stderr)
    if args.base_url:
        print(f"Base URL : {args.base_url}", file=sys.stderr)
    print(f"{sep}\n", file=sys.stderr)

    # ── Rewrite ───────────────────────────────────────────────────────────────
    shared = dict(
        provider_cfg=provider_cfg,
        model=model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    rewritten = rewrite_prompt(
        raw_prompt,
        domain=args.domain,
        **shared,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        save_version(args.save, raw_prompt, rewritten, args.domain, provider_name, model)

    # ── Compare ───────────────────────────────────────────────────────────────
    if args.compare:
        compare_prompts(raw_prompt, rewritten, args.sample_input, **shared)


if __name__ == "__main__":
    main()
