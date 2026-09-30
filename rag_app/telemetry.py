"""No-op ChromaDB product telemetry client.

The `posthog` package resolved in this project's environment has a newer,
incompatible `capture()` signature than the one ChromaDB's bundled telemetry
client (`chromadb.telemetry.product.posthog.Posthog`) expects. ChromaDB calls
``posthog.capture(user_id, event_name, properties)`` with three positional
arguments, but the installed ``posthog`` only accepts one positional argument
(the event name) plus keyword arguments. That mismatch raises a ``TypeError``
on every telemetry event, which ChromaDB swallows and logs as an ERROR
(e.g. "Failed to send telemetry event ...: capture() takes 1 positional
argument but 3 were given"). Setting ``anonymized_telemetry=False`` alone does
not prevent this, because the crash happens at Python's argument-binding step
before any "is telemetry enabled" check ever runs.

This module provides a drop-in replacement telemetry client that never calls
posthog at all, avoiding the crash (and any log noise) entirely. It is wired
up via the ``CHROMA_PRODUCT_TELEMETRY_IMPL`` environment variable in
:mod:`rag_app.vector_store`.
"""

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopProductTelemetryClient(ProductTelemetryClient):
    """A `ProductTelemetryClient` that discards every event without sending it."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
