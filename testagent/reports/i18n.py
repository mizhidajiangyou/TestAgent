"""
Lightweight i18n helpers for report generators.

Maps report section headers and enum values (test type / priority)
between English and Simplified Chinese.
"""

from typing import Literal

OutputLanguage = Literal["english", "chinese"]

_EN = "english"
_ZH = "chinese"

#: Section/field labels used by report generators.
REPORT_LABELS: dict[str, dict[str, str]] = {
    _EN: {
        "author": "Author",
        "date": "Date",
        "version": "Version",
        "total_test_cases": "Total Test Cases",
        "summary": "Summary",
        "by_type": "By Test Type",
        "by_priority": "By Priority",
        "type": "Type",
        "count": "Count",
        "priority": "Priority",
        "test_cases": "Test Cases",
        "endpoint": "Endpoint",
        "description": "Description",
        "preconditions": "Preconditions",
        "steps": "Steps",
        "expected_results": "Expected Results",
        "test_configuration": "Test Configuration",
        "parameter": "Parameter",
        "value": "Value",
        "test_results": "Test Results",
        "kpi": "Key Performance Indicators",
        "metric": "Metric",
        "endpoint_breakdown": "Endpoint Breakdown",
        "errors": "Errors",
        "ai_analysis": "AI Analysis",
        "verdict": "Verdict",
        "summary_label": "Summary",
        "findings": "Findings",
        "next_steps": "Recommended Next Steps",
        "how_to_run": "How to Run",
    },
    _ZH: {
        "author": "作者",
        "date": "日期",
        "version": "版本",
        "total_test_cases": "测试用例总数",
        "summary": "汇总",
        "by_type": "按测试类型",
        "by_priority": "按优先级",
        "type": "类型",
        "count": "数量",
        "priority": "优先级",
        "test_cases": "测试用例",
        "endpoint": "接口",
        "description": "描述",
        "preconditions": "前置条件",
        "steps": "操作步骤",
        "expected_results": "预期结果",
        "test_configuration": "测试配置",
        "parameter": "参数",
        "value": "值",
        "test_results": "测试结果",
        "kpi": "关键性能指标",
        "metric": "指标",
        "endpoint_breakdown": "接口明细",
        "errors": "错误数",
        "ai_analysis": "AI 分析",
        "verdict": "结论",
        "summary_label": "总结",
        "findings": "发现的问题",
        "next_steps": "建议后续步骤",
        "how_to_run": "如何运行",
    },
}

#: Test type value translations.
TYPE_LABELS: dict[str, dict[str, str]] = {
    _EN: {
        "functional": "functional",
        "boundary": "boundary",
        "negative": "negative",
        "performance": "performance",
        "security": "security",
        "integration": "integration",
    },
    _ZH: {
        "functional": "功能",
        "boundary": "边界",
        "negative": "负向",
        "performance": "性能",
        "security": "安全",
        "integration": "集成",
    },
}

#: Priority value translations.
PRIORITY_LABELS: dict[str, dict[str, str]] = {
    _EN: {"high": "high", "medium": "medium", "low": "low"},
    _ZH: {"high": "高", "medium": "中", "low": "低"},
}


def is_chinese(output_language: str) -> bool:
    """Return True when the target output language is Chinese."""
    return output_language == _ZH


def report_label(key: str, output_language: str) -> str:
    """Return localized report section label."""
    table = REPORT_LABELS.get(output_language, REPORT_LABELS[_EN])
    return table.get(key, key)


def type_label(value: str, output_language: str) -> str:
    """Return localized test type label."""
    table = TYPE_LABELS.get(output_language, TYPE_LABELS[_EN])
    return table.get(value, value)


def priority_label(value: str, output_language: str) -> str:
    """Return localized priority label."""
    table = PRIORITY_LABELS.get(output_language, PRIORITY_LABELS[_EN])
    return table.get(value, value)
