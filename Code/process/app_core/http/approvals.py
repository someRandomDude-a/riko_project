"""/api/tools/approvals: which tools ask first (tools/approval.py) and the user's answer to a pending request."""
from fastapi import APIRouter, Header, HTTPException

from .backend import Services

router = APIRouter()


@router.get('/api/tools/approvals')
def tool_approvals(backend: Services): return backend.approvals()

@router.put('/api/tools/approvals')
def approval_policy(request: dict, backend: Services, x_riko_confirmation: str | None = Header(default=None)):
    registry = backend.tool_registry()
    # Switching approval off lets the model run that tool unattended: confirm it, whether the tool asks by its own rule or
    # by its default (a configured MCP server's tools ask first unless mcp.json says not).
    policy, current = request.get('policy'), registry.approvals.snapshot(registry.tools)
    if isinstance(policy, dict):
        backend.require_confirmation('tool_approvals', {name: False for name, value in policy.items()
            if value is False and current['required'].get(name, True)}, x_riko_confirmation)
    try: return registry.approvals.configure(policy, registry.tools)
    except ValueError as exc: raise HTTPException(400, str(exc))
    except OSError as exc: raise HTTPException(409, f'Could not save tool permissions: {exc}')

@router.post('/api/tools/approvals/{request_id}')
def resolve_approval(request_id: str, request: dict, backend: Services):
    try: backend.tool_registry().approvals.resolve(request_id, request.get('approved'))
    except ValueError as exc: raise HTTPException(409, str(exc))
    return {'ok': True}
