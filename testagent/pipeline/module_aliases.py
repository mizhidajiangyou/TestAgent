"""R6 module alias table (LINK-S2a, plan-links-v15 §4.2).

CJK vocabulary lives in its own file (ruff RUF001/002/003 exemptions are
scoped here and only here — the lint gate remembers where the words are).
``aliases=None`` and ``{}`` both mean the built-in table; a non-empty dict
only MERGES additional entries. The single off switch is
``LINKS_PROSE_ENABLED=false``.
"""

from __future__ import annotations

__all__ = ["BUILTIN_ALIASES", "STRONG_SCORE", "WEAK_SCORE", "merge_aliases"]

STRONG_SCORE = 1.0
WEAK_SCORE = 0.5

#: module -> (strong aliases, weak aliases). English canonical keys are
#: strong aliases of themselves in singular AND plural forms.
BUILTIN_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "products": (("商品", "产品", "库存"), ()),
    "cart": (("购物车",), ()),
    "orders": (("订单",), ()),
    "users": (("用户", "账号", "账户"), ()),
    "payment": (("支付", "付款"), ()),
    "coupon": (("优惠券", "优惠码"), ("优惠",)),
    "logistics": (("物流", "配送", "运单"), ("发货",)),
    "notification": (("消息通知", "通知服务"), ("通知",)),
}


def merge_aliases(
    extra: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """None/{}/missing -> builtin; non-empty merges incrementally (an alias
    declared by two real modules is an ambiguity flagged by the matcher,
    never resolved by dict order)."""
    if not extra:
        return dict(BUILTIN_ALIASES)
    merged = {k: (tuple(s), tuple(w)) for k, (s, w) in BUILTIN_ALIASES.items()}
    for module, (strong, weak) in extra.items():
        key = module.strip().casefold()
        if key in merged:
            prev = merged[key]
            merged[key] = (
                tuple(dict.fromkeys(prev[0] + tuple(strong))),
                tuple(dict.fromkeys(prev[1] + tuple(weak))),
            )
        else:
            merged[key] = (tuple(strong), tuple(weak))
    return merged
