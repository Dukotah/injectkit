"""Target adapters: things injectkit can send attack prompts to.

Every adapter implements the :class:`~injectkit.targets.base.Target` protocol.
Concrete adapters (http, anthropic_target, mcp) lazy-import their heavy SDKs so
importing this package never requires anthropic/mcp/httpx to be installed.
"""

from __future__ import annotations

from .base import Target

__all__ = ["Target"]
