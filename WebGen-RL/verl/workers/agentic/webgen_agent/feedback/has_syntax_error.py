from pathlib import Path
from typing  import Tuple
import importlib.util as _iu
import json, re

# ----------------------------------------------------------------------
# Helper: detect whether a package can be imported
# ----------------------------------------------------------------------
def _has(pkg: str) -> bool:
    return _iu.find_spec(pkg) is not None

# ----------------------------------------------------------------------
# Helper: very light-weight balanced-bracket / quote check -------------
# (good enough to flag *gross* typos when the real parser is absent)
# ----------------------------------------------------------------------
_PAIRS = {"{": "}", "(": ")", "[": "]"}

def _quick_balance_scan(src: str) -> Tuple[bool, str]:
    stack: list[tuple[str, int]] = []
    quote: str | None = None
    escape = False

    for i, ch in enumerate(src):
        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if quote:
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            continue

        if ch in _PAIRS:
            stack.append((ch, i))
        elif ch in _PAIRS.values():
            if not stack or _PAIRS[stack[-1][0]] != ch:
                return False, f"Unmatched {ch!r} at position {i}"
            stack.pop()

    if quote:
        return False, f"Unterminated string starting with {quote!r}"
    if stack:
        opener, pos = stack[-1]
        return False, f"Missing {_PAIRS[opener]!r} to match {opener!r} at {pos}"
    return True, ""


# ----------------------------------------------------------------------
# Main: syntax checker --------------------------------------------------
# ----------------------------------------------------------------------
def has_syntax_error(
    filename: str,
    source: str,
    *,
    strict_missing_deps: bool = False,
) -> Tuple[bool, str]:
    """
    (error, message)

    • error == False   → file is syntactically OK **or** only a parser wheel is missing
    • error == True    → definite syntax error (or unknown extension / strict missing dep)

    Parameters
    ----------
    strict_missing_deps : bool, default False
        If True, the *absence* of the third-party parser is treated as an error.
        If False (default), we fall back to the quick scan shown above.
    """
    ext = Path(filename).suffix.lower()

    # ---------- JSON (stdlib = always present) -------------------------
    if ext == ".json":
        try:
            json.loads(source)
            return False, ""
        except Exception as exc:
            return True, str(exc)

    # ---------- HTML ---------------------------------------------------
    if ext in {".html", ".htm"}:
        if not _has("lxml"):
            if strict_missing_deps:
                return True, "lxml not installed"

            # --- NEW, SAFER TAG-BALANCE CHECK ---------------------------
            #  • keeps the leading '/' so we can tell opens from closes
            #  • skips <!DOCTYPE …> and other <! ... > declarations
            tag_pat  = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9:_-]+)[^>]*?>")
            voids    = {
                "area", "base", "br", "col", "embed", "hr", "img", "input",
                "link", "meta", "param", "source", "track", "wbr",
            }

            stack: list[str] = []
            for m in tag_pat.finditer(source):
                slash, name = m.groups()
                name = name.lower()

                # ignore declarations like <!DOCTYPE html>
                if name.startswith("!"):
                    continue

                if slash:                       # closing tag
                    if not stack or stack[-1] != name:
                        return True, f"Mismatched </{name}> at char {m.start()}"
                    stack.pop()
                else:                           # opening tag
                    if name not in voids:
                        stack.append(name)

            if stack:
                return True, f"Unclosed <{stack[-1]}> tag"

            return False, "Skipped full HTML parse (lxml absent)"
        else:
            from lxml import etree as ET
            try:
                ET.fromstring(source.encode(), parser=ET.HTMLParser(recover=False))
                return False, ""
            except Exception as exc:
                return True, str(exc)

    # ---------- CSS ----------------------------------------------------
    if ext == ".css":
        if _has("tinycss2"):
            import tinycss2
            tokens = tinycss2.parse_stylesheet_bytes(
                source.encode(), skip_comments=True, skip_whitespace=True
            )
            errors = [t for t in tokens if t.type == "error"]
            return (True, errors[0].message) if errors else (False, "")
        elif strict_missing_deps:
            return True, "tinycss2 not installed"
        else:
            ok, msg = _quick_balance_scan(source)
            return (not ok, msg or "Skipped full CSS parse (tinycss2 absent)")

    # ---------- JS / JSX / TS / TSX ------------------------------------
    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        if _has("esprima"):
            import esprima
            opts = {"jsx": ext in {".jsx", ".tsx"}}
            try:
                if ext == ".js":
                    esprima.parseScript(source, tolerant=False, **opts)
                else:
                    esprima.parseModule(source, tolerant=False, **opts)
                return False, ""
            except Exception as exc:
                return True, str(exc)
        elif strict_missing_deps:
            return True, "esprima not installed"
        else:
            ok, msg = _quick_balance_scan(source)
            return (not ok, msg or "Skipped full JS/TS parse (esprima absent)")

    return True, f"Unsupported extension: {ext}"


if __name__ == "__main__":
    # Example usage
    import json
    in_file = "/mnt/cache/agent/Zimu/WebGen-Agent/workspaces_root/debug/000002/index.html"
    with open(in_file, "r", encoding="utf-8") as f:
        content = f.read()
    has_error, explanation = has_syntax_error(in_file, content)
    if has_error:
        print(f"Syntax error in {in_file}: {explanation}")
    else:
        print(f"No syntax error in {in_file}.")