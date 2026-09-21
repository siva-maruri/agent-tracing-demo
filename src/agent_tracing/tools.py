"""Two tools over fake support data. The order record includes a customer email on
purpose: it's the kind of thing that leaks into traces through tool results."""

from langchain_core.tools import tool

_ORDERS = {
    "1042": {
        "order_id": "1042",
        "customer_email": "jane.doe@contoso.com",
        "amount": 59.90,
        "currency": "USD",
        "charge_count": 2,
        "status": "shipped",
    },
}

_REFUNDS = {
    "1042": {"refund_id": "rf_77310", "amount": 59.90, "issued_on": "2026-09-18", "state": "processing"},
}


class NotFound(LookupError):
    pass


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by id: amount, number of charges, shipping status."""
    try:
        return _ORDERS[order_id]
    except KeyError:
        raise NotFound(f"no order {order_id}") from None


@tool
def refund_status(order_id: str) -> dict:
    """Check whether a refund exists for an order and where it is."""
    return _REFUNDS.get(order_id, {"state": "none"})


TOOLS = [lookup_order, refund_status]
