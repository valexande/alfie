"""Dependency-free, human-readable HTML rendering for bias reports."""

from __future__ import annotations

from html import escape
from typing import Any


def _format(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return escape(str(value))


def _percent(value: Any) -> str:
    return "Not available" if value is None else f"{float(value) * 100:.1f}%"


def _card(label: str, value: Any, hint: str = "", tone: str = "") -> str:
    return (
        f'<div class="metric-card {tone}"><span>{escape(label)}</span>'
        f"<strong>{_format(value)}</strong><small>{escape(hint)}</small></div>"
    )


def _status_message(data: dict[str, Any]) -> str:
    status = data.get("status")
    if status in {"not_run", "not_applicable"}:
        reason = data.get("reason", "This analysis does not apply to the supplied data.")
        return f'<p class="muted-box">{escape(reason)}</p>'
    return ""


def _render_group_analysis(analysis: dict[str, Any]) -> str:
    rows = []
    for group in analysis["groups"]:
        performance = group.get("performance", {})
        rows.append(
            "<tr>"
            f"<td>{escape(', '.join(f'{key}={value}' for key, value in group['group'].items()))}</td>"
            f"<td>{group['sample_count']}</td>"
            f"<td>{_percent(group['sample_share'])}</td>"
            f"<td>{_percent(group.get('selection_rate'))}</td>"
            f"<td>{_percent(performance.get('accuracy'))}</td>"
            f"<td>{_percent(performance.get('true_positive_rate'))}</td>"
            f"<td>{_percent(performance.get('false_positive_rate'))}</td>"
            "</tr>"
        )
    disparity_cards = "".join(
        _card(key.replace("_", " ").title(), value)
        for key, value in analysis["disparities"].items()
    )
    uncertainty = analysis.get("uncertainty", {}).get(
        "demographic_parity_difference", {}
    )
    interval = ""
    if uncertainty.get("ci_low") is not None:
        interval = (
            '<p class="explanation"><strong>Uncertainty:</strong> The estimated '
            f"selection-rate difference is {_format(uncertainty['estimate'])}; its "
            f"95% bootstrap interval is {_format(uncertainty['ci_low'])}–"
            f"{_format(uncertainty['ci_high'])}.</p>"
        )
    flags = "".join(
        f'<li class="flag">{escape(message)}</li>'
        for message in analysis["review_flags"]
    )
    flag_block = f"<ul class=\"flag-list\">{flags}</ul>" if flags else (
        '<p class="pass">No configured screening threshold was crossed.</p>'
    )
    return (
        f'<section><div class="section-heading"><div><p class="eyebrow">'
        f"{escape(analysis['analysis_type'].replace('_', ' '))}</p>"
        f"<h2>{escape(' × '.join(analysis['attributes']))}</h2></div>"
        f"<span class=\"count\">{analysis['rows_analyzed']} rows</span></div>"
        f'<div class="metric-grid">{disparity_cards}</div>{interval}'
        '<div class="table-wrap"><table><thead><tr><th>Group</th><th>N</th>'
        "<th>Dataset share</th><th>Positive decisions</th><th>Accuracy</th>"
        "<th>True-positive rate</th><th>False-positive rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{flag_block}</section>"
    )


def _render_split_distribution(data: dict[str, Any]) -> str:
    unavailable = _status_message(data)
    if unavailable:
        return unavailable
    shares = data.get("group_shares_by_split", {})
    groups = sorted({group for split in shares.values() for group in split})
    rows = []
    for split, values in shares.items():
        rows.append(
            f"<tr><td>{escape(str(split))}</td>"
            + "".join(f"<td>{_percent(values.get(group))}</td>" for group in groups)
            + "</tr>"
        )
    return (
        f'<p class="explanation">Compares <strong>{escape(data["sensitive_attribute"])}</strong> '
        f'representation across values of <strong>{escape(data["split_column"])}</strong>. '
        "Large changes can make validation results unrepresentative.</p>"
        '<div class="table-wrap"><table><thead><tr><th>Split</th>'
        + "".join(f"<th>{escape(group)}</th>" for group in groups)
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_data_quality(data: dict[str, list[dict[str, Any]]]) -> str:
    sections = []
    for attribute, groups in data.items():
        rows = "".join(
            "<tr>"
            f"<td>{escape(group['group'])}</td><td>{group['sample_count']}</td>"
            f"<td>{_percent(group['sample_share'])}</td>"
            f"<td>{_percent(group['mean_missing_value_rate'])}</td>"
            f"<td>{_percent(group.get('observed_positive_label_rate'))}</td></tr>"
            for group in groups
        )
        sections.append(
            f"<h4>{escape(attribute)}</h4><div class=\"table-wrap\"><table><thead><tr>"
            "<th>Group</th><th>Samples</th><th>Dataset share</th>"
            "<th>Average missingness</th><th>Observed positive-label rate</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return "".join(sections)


def _render_proxies(data: dict[str, dict[str, Any]]) -> str:
    sections = []
    for attribute, result in data.items():
        unavailable = _status_message(result)
        if unavailable:
            sections.append(f"<h4>{escape(attribute)}</h4>{unavailable}")
            continue
        score = result.get("balanced_accuracy", result.get("maximum_proxy_association"))
        tone = "warn" if result.get("review_recommended") else "good"
        rows = "".join(
            f"<tr><td>{escape(item['feature'])}</td>"
            f"<td>{_format(item['normalized_mutual_information'])}</td></tr>"
            for item in result.get("top_proxy_features", [])
        )
        table = (
            '<div class="table-wrap"><table><thead><tr><th>Feature</th>'
            f"<th>Association</th></tr></thead><tbody>{rows}</tbody></table></div>"
            if rows
            else ""
        )
        sections.append(
            f"<h4>{escape(attribute)}</h4><div class=\"metric-grid\">"
            f"{_card('Strongest proxy signal', score, tone=tone)}"
            f"{_card('Review recommended', result.get('review_recommended'), tone=tone)}"
            f"</div>{table}<p class=\"caption\">{escape(result.get('interpretation', ''))}</p>"
        )
    return "".join(sections)


def _render_thresholds(data: dict[str, Any]) -> str:
    unavailable = _status_message(data)
    if unavailable:
        return unavailable
    rows = []
    for item in data["thresholds"]:
        difference = item["demographic_parity_difference"]
        width = min(100, max(0, float(difference or 0) * 100))
        rows.append(
            f"<tr><td>{item['threshold']:.1f}</td>"
            f"<td><div class=\"bar-track\"><span style=\"width:{width:.1f}%\"></span></div>"
            f"{_format(difference)}</td>"
            f"<td>{_format(item['disparate_impact_ratio'])}</td></tr>"
        )
    return (
        f'<p class="explanation">Shows how fairness indicators change when the decision '
        f'threshold moves. Attribute analyzed: <strong>{escape(data["sensitive_attribute"])}</strong>. '
        "A model may look different at 0.5 than at the threshold used in production.</p>"
        '<div class="table-wrap"><table><thead><tr><th>Score threshold</th>'
        "<th>Selection-rate difference</th><th>Impact ratio</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_modality(data: dict[str, Any]) -> str:
    parts = []
    text = data.get("text_counterfactual", {})
    if text.get("status") == "complete":
        parts.append(
            "<h3>Text counterfactual test</h3>"
            '<p class="explanation">Compares paired records that retain the same business '
            "information while changing only the tested identity reference.</p>"
            '<div class="metric-grid">'
            f"{_card('Pairs tested', text.get('pairs_analyzed'))}"
            f"{_card('Prediction flip rate', _percent(text.get('counterfactual_flip_rate')), tone='warn')}"
            f"{_card('Mean score change', text.get('mean_absolute_score_difference'))}"
            f"{_card('Maximum score change', text.get('maximum_score_difference'))}"
            "</div>"
        )
    for key, title in (("vision", "Vision diagnostics"), ("video", "Video diagnostics")):
        result = data.get(key, {})
        if result.get("status") == "complete":
            cards = "".join(
                _card(name.replace("_", " ").title(), value)
                for name, value in result.items()
                if name not in {"status", "group_metrics", "vision_task"}
                and not isinstance(value, (dict, list))
            )
            parts.append(f"<h3>{title}</h3><div class=\"metric-grid\">{cards}</div>")
    return "".join(parts) or '<p class="muted-box">No modality-specific analysis applies.</p>'


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    flags = report.get("flags", [])
    flag_items = "".join(
        f'<li class="flag"><strong>{escape(" × ".join(item["attributes"]))}:</strong> '
        f'{escape(item["message"])}</li>'
        for item in flags
    )
    overview = "".join(
        [
            _card("Rows analyzed", summary["rows"]),
            _card("Sensitive attributes", len(summary["sensitive_attributes"])),
            _card("Analyses completed", summary["analyses_run"]),
            _card("Review flags", summary["flags_count"], tone="warn" if flags else "good"),
        ]
    )
    analysis_sections = "".join(
        _render_group_analysis(analysis) for analysis in report["analyses"]
    )
    diagnostics = report.get("dataset_diagnostics", {})
    limitations = "".join(
        f"<li>{escape(item)}</li>" for item in report["methodology"]["limitations"]
    )
    status_class = "review" if report["status"] == "review_recommended" else "clear"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AutoML Bias Detection Report</title>
<style>
:root{{--ink:#172033;--navy:#193b69;--blue:#2e67a5;--line:#d8dee9;--soft:#f5f8fc;
--amber:#a76500;--amber-bg:#fff4d6;--green:#216e45;--green-bg:#eaf7ef;}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef3f8;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;background:white;min-height:100vh;padding:48px 56px}}
h1{{font-size:36px;margin:.2rem 0}} h2{{color:var(--navy);font-size:25px;margin:0}} h3{{color:var(--navy);margin-top:26px}}
h4{{margin:24px 0 8px}} .eyebrow{{margin:0;color:var(--blue);font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.12em}}
.hero{{border-bottom:1px solid var(--line);padding-bottom:30px}} .subtitle{{max-width:800px;color:#526173}}
.status{{display:inline-block;padding:6px 11px;border-radius:20px;font-weight:700}} .status.review{{color:#7b4800;background:var(--amber-bg)}}
.status.clear{{color:var(--green);background:var(--green-bg)}} section{{padding:32px 0;border-bottom:1px solid var(--line)}}
.section-heading{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}} .count{{color:#657386}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:18px 0}}
.metric-card{{border:1px solid var(--line);border-radius:10px;padding:15px;background:var(--soft)}}
.metric-card span,.metric-card small{{display:block;color:#657386}} .metric-card strong{{display:block;font-size:22px;color:var(--navy);margin:4px 0}}
.metric-card.warn{{background:var(--amber-bg);border-color:#ebca83}} .metric-card.good{{background:var(--green-bg);border-color:#b6ddc6}}
.notice{{background:var(--amber-bg);border-left:5px solid #e0a100;padding:14px 18px;margin:22px 0}}
.explanation,.caption{{color:#526173}} .caption{{font-size:13px}} .table-wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border-bottom:1px solid var(--line);padding:10px;text-align:left}}
th{{background:#edf3fa;color:#34465d}} .flag-list{{padding:0;list-style:none}} .flag{{background:var(--amber-bg);padding:10px 13px;margin:7px 0;border-radius:7px}}
.pass{{color:var(--green);background:var(--green-bg);padding:10px 13px;border-radius:7px}}
.muted-box{{padding:13px;background:var(--soft);color:#657386;border-radius:7px}} .bar-track{{display:inline-block;width:120px;height:8px;background:#dfe7f0;border-radius:5px;margin-right:10px;vertical-align:middle}}
.bar-track span{{display:block;height:100%;background:var(--blue);border-radius:5px}}
.glossary dt{{font-weight:750;color:var(--navy);margin-top:10px}} .glossary dd{{margin-left:0;color:#526173}}
@media(max-width:700px){{main{{padding:28px 18px}} h1{{font-size:29px}} .section-heading{{display:block}}}}
</style></head><body><main>
<header class="hero"><p class="eyebrow">ALFIE trustworthiness reporting</p>
<h1>AutoML Bias Detection Report</h1>
<p class="subtitle">Group-level screening of model outcomes, performance, dataset representation,
probability thresholds, and modality-specific behavior.</p>
<span class="status {status_class}">{escape(report["status"].replace("_", " ").title())}</span>
<p class="notice"><strong>Important:</strong> This report identifies signals for human review.
It does not establish that a model is fair, unfair, or legally discriminatory.</p>
<div class="metric-grid">{overview}</div>
<p><strong>Task:</strong> {escape(summary["task_type"].replace("_", " "))} ·
<strong>Modality:</strong> {escape(summary["modality"].replace("_", " + "))} ·
<strong>Positive outcome:</strong> {escape(summary["positive_label"])}</p></header>
<section><div class="section-heading"><div><p class="eyebrow">Executive summary</p>
<h2>Signals requiring review</h2></div></div>
{f'<ul class="flag-list">{flag_items}</ul>' if flags else '<p class="pass">No configured screening threshold was crossed.</p>'}
</section>
{analysis_sections}
<section><p class="eyebrow">Data coverage</p><h2>Train/test representation</h2>
{_render_split_distribution(diagnostics.get("split_distribution", {}))}</section>
<section><p class="eyebrow">Data quality</p><h2>Representation, labels and missingness</h2>
<p class="explanation">Shows whether groups have enough representation, different observed-label
rates, or systematically different missing-data rates.</p>
{_render_data_quality(diagnostics.get("group_data_quality", {}))}</section>
<section><p class="eyebrow">Proxy screening</p><h2>Can other columns reveal a sensitive attribute?</h2>
<p class="explanation">High association suggests that removing the sensitive column alone may not
prevent the model from reconstructing sensitive-group information.</p>
{_render_proxies(diagnostics.get("proxy_predictability", {}))}</section>
<section><p class="eyebrow">Decision policy</p><h2>Probability threshold sensitivity</h2>
{_render_thresholds(report.get("score_threshold_analysis", {}))}</section>
<section><p class="eyebrow">Modality checks</p><h2>Text, image or video diagnostics</h2>
{_render_modality(report.get("modality_analysis", {}))}</section>
<section><p class="eyebrow">Reference</p><h2>How to read the metrics</h2>
<dl class="glossary"><dt>Selection-rate difference</dt><dd>Gap between the highest and lowest
group positive-decision rates. Smaller is usually better, subject to context.</dd>
<dt>Disparate-impact ratio</dt><dd>Lowest divided by highest positive-decision rate. Values below
0.80 trigger review in this report; this is a screening convention, not a verdict.</dd>
<dt>Equal opportunity difference</dt><dd>Gap in true-positive rates: how often qualified positive
cases are correctly recognized across groups.</dd><dt>Equalized odds difference</dt><dd>The larger
of the true-positive-rate and false-positive-rate gaps.</dd><dt>Counterfactual flip rate</dt>
<dd>Percentage of paired text cases whose predicted label changes after only the tested identity
reference changes.</dd></dl></section>
<section><p class="eyebrow">Limitations</p><h2>Required human interpretation</h2>
<ul>{limitations}</ul></section>
</main></body></html>"""
