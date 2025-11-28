# AGENTS.md - Photo Brain

Authoritative rules for all coding agents (VS Code, Copilot, Codex).

This repo uses strict module boundaries, typed models, and agent-friendly contracts.  
Agents MUST follow the rules in:

- docs/AGENT_RULES.md  
- docs/ARCHITECTURE.md  

The full coding rules, module contracts, and allowed dependencies are defined there.

## Agent Expectations

- Keep PRs minimal, clean, and bounded.
- Follow architecture.  
- Use core models only.  
- Never bypass vector backend or DB layers.
- Pass tests.

See more in docs/AGENT_RULES.md.
