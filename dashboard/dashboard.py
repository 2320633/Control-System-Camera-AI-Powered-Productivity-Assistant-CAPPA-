#!/usr/bin/env python3
"""Generate the repository's README dashboard and compact SVG summary."""

import argparse
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
CONFIG_PATH = DASHBOARD / "config.json"
TEMPLATE_PATH = DASHBOARD / "README_TEMPLATE.md"
README_PATH = ROOT / "README.md"
SVG_PATH = DASHBOARD / "assets" / "repository-summary.svg"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".swift",
    ".ts",
    ".tsx",
}
LANGUAGE_NAMES = {
    ".ipynb": "Jupyter notebooks",
    ".md": "Markdown",
    ".py": "Python",
    ".txt": "Text",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
}


def run_git(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def project_files(config):
    excluded = set(config["excluded_directories"])
    files = []
    for current, directories, names in os.walk(str(ROOT), topdown=True):
        directories[:] = sorted(
            name
            for name in directories
            if name not in excluded and (not name.startswith(".") or name == ".github")
        )
        for name in names:
            if name.startswith(".") or name.startswith("CAPA_Session_Report_"):
                continue
            files.append(Path(current) / name)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix().casefold())


def source_files(files):
    return [
        path
        for path in files
        if path.relative_to(ROOT).parts[0] != "dashboard"
        and (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix.lower() == ".ipynb")
    ]


def read_sources(paths):
    contents = []
    for path in paths:
        try:
            contents.append(path.read_text(encoding="utf-8").casefold())
        except UnicodeDecodeError:
            continue
    return "\n".join(contents)


def dependency_names():
    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        return []
    names = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith(("-", "#")):
            continue
        name = re.split(r"\s*(?:===|==|~=|!=|<=|>=|<|>|;|\[)", requirement, maxsplit=1)[0]
        if name:
            names.add(name.casefold())
    return sorted(names)


def git_activity():
    commits = run_git("rev-list", "--count", "HEAD")
    recent = run_git("log", "-5", "--format=%cs%x09%s")
    latest_date = run_git("log", "-1", "--format=%cs")
    latest_subject = run_git("log", "-1", "--format=%s")
    remote = run_git("config", "--get", "remote.origin.url") or ""

    repository = "Local repository"
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", remote)
    if match:
        repository = match.group(1)

    recent_commits = []
    if recent:
        for line in recent.splitlines():
            date, separator, subject = line.partition("\t")
            if separator:
                recent_commits.append((date, subject))

    return {
        "commit_count": commits or "0",
        "latest_date": latest_date or "Unavailable",
        "latest_subject": latest_subject or "No commits found",
        "recent_commits": recent_commits,
        "repository": repository,
    }


def has_tests(files):
    for path in files:
        relative = path.relative_to(ROOT)
        parts = {part.casefold() for part in relative.parts}
        name = path.name.casefold()
        if "test" in parts or "tests" in parts:
            return True
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
    return False


def has_test_workflow(files):
    for path in files:
        relative = path.relative_to(ROOT).as_posix().casefold()
        if not relative.startswith(".github/workflows/") or path.suffix not in {".yml", ".yaml"}:
            continue
        content = path.read_text(encoding="utf-8").casefold()
        if re.search(r"\b(pytest|unittest|nose2)\b", content):
            return True
    return False


def has_evaluation_assets(files):
    return any(
        re.search(r"(?:eval|benchmark|dataset|ground.?truth)", path.as_posix(), re.IGNORECASE)
        for path in files
    )


def has_calibration_guidance(files):
    for path in files:
        if path.suffix.lower() not in {".md", ".py", ".ipynb"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative == "README.md" or relative.startswith("dashboard/"):
            continue
        if re.search(r"\bcalibrat(?:e|ed|es|ing|ion)\b", path.read_text(encoding="utf-8"), re.IGNORECASE):
            return True
    return False


def detected_features(config, corpus):
    return [
        feature
        for feature in config["features"]
        if any(marker.casefold() in corpus for marker in feature["evidence"])
    ]


def render_features(features):
    if not features:
        return "- No implementation features could be inferred from the current source files."
    return "\n".join(
        "- **{}** — {}".format(feature["title"], feature["description"])
        for feature in features
    )


def render_achievements(features, report_count, activity):
    if features:
        names = ", ".join(feature["title"] for feature in features)
        implementation = "- **{} code-backed capabilities detected:** {}.".format(
            len(features), names
        )
    else:
        implementation = "- No code-backed capabilities have been detected yet."
    if report_count:
        reports = "- **{} session report examples present**; this dashboard counts filenames only.".format(
            report_count
        )
    else:
        reports = "- No session report examples are present yet."
    latest = "- **Latest recorded change:** {} — {}.".format(
        activity["latest_date"],
        activity["latest_subject"].replace("|", "\\|").replace("`", "'"),
    )
    return "\n".join((implementation, reports, latest))


def render_roadmap(config, files):
    checks = {
        "tests": has_tests(files),
        "evaluation": has_evaluation_assets(files),
        "calibration": has_calibration_guidance(files),
        "test_workflow": has_test_workflow(files),
    }
    remaining = [
        "- **{}** — {}".format(item["title"], item["description"])
        for item in config["roadmap"]
        if not checks.get(item["completion_check"], False)
    ]
    if not remaining:
        return "- All currently tracked roadmap checks are complete. Review the source and refresh these milestones as the project evolves."
    return "\n".join(remaining)


def render_structure(files):
    selected = []
    for path in files:
        relative = path.relative_to(ROOT)
        if len(relative.parts) > 3:
            continue
        if relative.parts[0] in {"dashboard", ".github"}:
            selected.append(relative)
        elif len(relative.parts) == 1:
            selected.append(relative)
    entries = []
    for relative in selected[:20]:
        label = relative.as_posix()
        entries.append("- [`{}`](./{})".format(label, label))
    if len(selected) > 20:
        entries.append("- Additional project files are available in the repository.")
    return "\n".join(entries) if entries else "- No project files found."


def language_summary(files):
    counts = {}
    for path in files:
        name = "Python dependencies" if path.name == "requirements.txt" else LANGUAGE_NAMES.get(path.suffix.lower())
        if name:
            counts[name] = counts.get(name, 0) + 1
        elif path.suffix.lower() in SOURCE_SUFFIXES:
            counts[path.suffix[1:].upper()] = counts.get(path.suffix[1:].upper(), 0) + 1
    if not counts:
        return "Not detected"
    return ", ".join(
        "{} ({})".format(name, count)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def render_recent_commits(commits):
    if not commits:
        return "- No commit history is available."
    return "\n".join(
        "- `{}` — {}".format(date, subject.replace("|", "\\|").replace("`", "'"))
        for date, subject in commits
    )


def build_dashboard(config, now):
    files = project_files(config)
    code_files = source_files(files)
    corpus = read_sources(code_files)
    features = detected_features(config, corpus)
    dependencies = dependency_names()
    activity = git_activity()
    python_lines = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in files
        if path.suffix.lower() == ".py"
    )
    report_count = sum(1 for _ in ROOT.glob("CAPA_Session_Report_*.txt"))
    repo_url = "https://github.com/{}".format(activity["repository"])
    clone_url = repo_url + ".git"
    clone_directory = activity["repository"].rsplit("/", 1)[-1]

    values = {
        "TITLE": config["title"],
        "TAGLINE": config["tagline"],
        "OVERVIEW": config["overview"],
        "REPOSITORY": activity["repository"],
        "REPOSITORY_URL": repo_url,
        "FEATURES": render_features(features),
        "ACHIEVEMENTS": render_achievements(features, report_count, activity),
        "PYTHON_LINES": "{:,}".format(python_lines),
        "DEPENDENCY_COUNT": str(len(dependencies)),
        "REPORT_COUNT": str(report_count),
        "PROJECT_FILES": "{:,}".format(len(files)),
        "COMMIT_COUNT": "{:,}".format(int(activity["commit_count"])),
        "LANGUAGES": language_summary(files),
        "PROJECT_STRUCTURE": render_structure(files),
        "ROADMAP": render_roadmap(config, files),
        "LATEST_COMMIT": activity["latest_subject"].replace("|", "\\|").replace("`", "'"),
        "LATEST_COMMIT_DATE": activity["latest_date"],
        "RECENT_COMMITS": render_recent_commits(activity["recent_commits"]),
        "GENERATED_DATE": now.strftime("%Y-%m-%d"),
        "CLONE_URL": clone_url,
        "CLONE_DIRECTORY": clone_directory,
    }
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    readme = template
    for key, value in values.items():
        readme = readme.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", readme):
        raise ValueError("The README template contains an unrendered placeholder.")

    metrics = [
        ("Python lines", values["PYTHON_LINES"]),
        ("Dependencies", values["DEPENDENCY_COUNT"]),
        ("Session reports", values["REPORT_COUNT"]),
        ("Git commits", values["COMMIT_COUNT"]),
        ("Project files", values["PROJECT_FILES"]),
    ]
    svg = render_svg(config["title"], config["tagline"], metrics, values["GENERATED_DATE"])
    return readme, svg


def render_svg(title, tagline, metrics, generated_date):
    safe_title = html.escape(title)
    safe_tagline = html.escape(tagline)
    cards = []
    for index, (label, value) in enumerate(metrics):
        x = 48 + index * 224
        cards.append(
            '<rect x="{}" y="164" width="208" height="96" rx="14" fill="#13253f" stroke="#29476e"/>'
            '<text x="{}" y="198" fill="#9db4d0" font-size="13">{}</text>'
            '<text x="{}" y="235" fill="#f2f7ff" font-size="25" font-weight="700">{}</text>'.format(
                x,
                x + 18,
                html.escape(label),
                x + 18,
                html.escape(value),
            )
        )
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="300" viewBox="0 0 1200 300" role="img" aria-labelledby="title description">
<title id="title">{}</title>
<desc id="description">{}</desc>
<defs><linearGradient id="background" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#091426"/><stop offset="1" stop-color="#132943"/></linearGradient></defs>
<rect width="1200" height="300" rx="22" fill="url(#background)"/>
<circle cx="1132" cy="55" r="120" fill="#2f7ca8" opacity=".14"/>
<text x="52" y="68" fill="#69d5c6" font-size="13" font-family="Segoe UI,Arial,sans-serif" font-weight="700" letter-spacing="2">REPOSITORY OVERVIEW</text>
<text x="52" y="112" fill="#f2f7ff" font-size="29" font-family="Segoe UI,Arial,sans-serif" font-weight="700">{}</text>
<text x="52" y="140" fill="#b4c4d8" font-size="15" font-family="Segoe UI,Arial,sans-serif">{}</text>
<g font-family="Segoe UI,Arial,sans-serif">{}</g>
<text x="1150" y="285" text-anchor="end" fill="#8098b5" font-size="11" font-family="Segoe UI,Arial,sans-serif">Generated {} UTC</text>
</svg>
""".format(safe_title, safe_tagline, safe_title, safe_tagline, "".join(cards), generated_date)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated README or SVG is out of date, without writing files",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    readme, svg = build_dashboard(config, datetime.now(timezone.utc))
    expected = {README_PATH: readme, SVG_PATH: svg}
    changed = []
    for path, content in expected.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != content:
            changed.append(path)

    if args.check:
        if changed:
            print(
                "Dashboard is out of date: {}".format(
                    ", ".join(path.relative_to(ROOT).as_posix() for path in changed)
                ),
                file=sys.stderr,
            )
            return 1
        print("Dashboard is up to date.")
        return 0

    for path in changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(expected[path])
        print("Updated {}".format(path.relative_to(ROOT).as_posix()))
    if not changed:
        print("Dashboard content is unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
