# Runtime Contract

The OVAgent harness has four durable runtime assets:

- `HarnessEvent`: normalized stream event emitted by the agent loop
- `StepTrace`: one structured execution step
- `RunTrace`: one end-to-end agent turn
- `ArtifactRef`: a durable pointer to code, notebooks, files, or reports

## Invariants

- `smart_agent.py` emits harness events.
- `omicverse_web/services/agent_service.py` forwards harness events without redefining their schema.
- Session history stores trace identifiers and summaries, not only free-form text.
- Replay and cleanup operate on stored `RunTrace` files.
