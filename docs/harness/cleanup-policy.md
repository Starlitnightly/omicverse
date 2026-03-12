# Cleanup Policy

Recurring cleanup should look for:

- duplicate runtime contracts between core and web
- stale execution-plan documents outside the harness index
- empty or orphaned trace files
- drift between tool registry, dispatcher, and generated docs

The first implementation only needs to emit reports. Automatic code changes are
explicitly out of scope for now.
