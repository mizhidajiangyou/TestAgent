"""S2a tests: clause splitting, alias scoring, direction templates (§4.2-4.3)."""

from testagent.config.models import APIEndpoint
from testagent.pipeline.links_r6 import (
    extract_r6_edges,
    ground_r6_edge,
    match_modules,
    split_clauses,
)


class TestClauses:
    def test_split_keeps_commas_and_offsets(self) -> None:
        text = "用户下单。订单创建时使用优惠券，然后通知物流！"
        clauses = split_clauses(text, "req")
        assert [c.text for c in clauses] == ["用户下单", "订单创建时使用优惠券，然后通知物流"]
        assert clauses[0].clause_id == "req-C001"

    def test_ascii_and_newline_separators(self) -> None:
        clauses = split_clauses("First clause. Second;\nthird?fourth!")
        assert len(clauses) == 4


class TestModuleMatching:
    def test_builtin_chinese_strong(self) -> None:
        clauses = split_clauses("购物车清空在订单生成之后执行")
        scores = match_modules(clauses[0])
        assert scores["cart"] >= 1.0
        assert scores["orders"] >= 1.0

    def test_weak_alias_counts_half(self) -> None:
        clauses = split_clauses("优惠活动相关内容")
        scores = match_modules(clauses[0], min_score=0.5)
        assert scores.get("coupon") == 0.5

    def test_english_plural_word_boundary(self) -> None:
        clauses = split_clauses("When an order is placed, notify logistics")
        scores = match_modules(clauses[0])
        assert "orders" in scores
        assert "logistics" in scores

    def test_below_min_score_not_recalled(self) -> None:
        clauses = split_clauses("优惠信息")
        assert match_modules(clauses[0], min_score=1.0) == {}
        assert "coupon" in match_modules(clauses[0], min_score=0.5)


class TestDirectionTemplates:
    def test_target_uses_source(self) -> None:
        """订单创建时使用优惠券 -> coupon -> orders (v15 §4.3 required
        positive)."""
        edges, abstained = extract_r6_edges("订单创建时使用优惠券")
        assert not abstained or all(a["reason"] != "no_template_match" for a in abstained)
        edge = next((e for e in edges if e.kind == "DATA_FLOW"), None)
        assert edge is not None
        assert (edge.source, edge.target) == ("coupon", "orders")

    def test_target_after_source(self) -> None:
        """购物车清空在订单生成之后执行 -> orders -> cart (v15 §4.3
        required positive; NOT cart->orders)."""
        edges, _ = extract_r6_edges("购物车清空在订单生成之后执行")
        edge = next((e for e in edges if e.kind == "SIDE_EFFECT"), None)
        assert edge is not None
        assert (edge.source, edge.target) == ("orders", "cart")

    def test_precondition_must_first(self) -> None:
        edges, _ = extract_r6_edges("支付必须先校验优惠券")
        assert edges and edges[0].kind == "PRECONDITION"
        assert (edges[0].source, edges[0].target) == ("coupon", "payment")

    def test_then_side_effect(self) -> None:
        edges, _ = extract_r6_edges("订单生成，随后更新物流单")
        assert edges and edges[0].kind == "SIDE_EFFECT"
        assert (edges[0].source, edges[0].target) == ("orders", "logistics")

    def test_negation_abstains(self) -> None:
        edges, abstained = extract_r6_edges("订单创建不依赖优惠券")
        assert edges == []
        assert any(a["reason"] == "unsupported_negation" for a in abstained)

    def test_three_modules_ambiguous(self) -> None:
        edges, abstained = extract_r6_edges("用户下单后订单使用优惠券并通知物流")
        assert all(e.kind for e in edges)  # no crash
        assert any(a["reason"] == "r6_ambiguous_direction" for a in abstained)


class TestGrounding:
    def _modules(self):
        return {
            "products": [
                APIEndpoint(method="POST", path="/products"),
                APIEndpoint(method="GET", path="/products/{id}"),
            ],
            "cart": [
                APIEndpoint(method="POST", path="/cart/items"),
                APIEndpoint(method="GET", path="/cart"),
            ],
        }

    def test_role_preference_order(self) -> None:
        from testagent.pipeline.links_r6 import R6Edge

        edge = R6Edge(
            source="products",
            target="cart",
            kind="DATA_FLOW",
            clause_id="C1",
            evidence="订单创建时使用优惠券",
            score=2.0,
        )
        pairs = ground_r6_edge(edge, self._modules())
        # write->write scores highest: POST /products -> POST /cart/items
        assert pairs[0] == ("POST /products", "POST /cart/items")
        assert len(pairs) <= 2

    def test_ungrounded_when_module_missing(self) -> None:
        from testagent.pipeline.links_r6 import R6Edge

        edge = R6Edge(
            source="payment",
            target="cart",
            kind="DATA_FLOW",
            clause_id="C1",
            evidence="x",
            score=2.0,
        )
        assert ground_r6_edge(edge, self._modules()) == []
