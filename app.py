"""
Root application entrypoint for Render and cloud deployments.
Forwards ASGI app from loanapp.backend.app
"""
from loanapp.backend.app import app

__all__ = ["app"]
