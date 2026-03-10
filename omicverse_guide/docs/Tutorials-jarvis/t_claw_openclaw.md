---
title: OpenClaw Integration with OmicVerse Claw
---

# OpenClaw Integration with OmicVerse Claw

OpenClaw integration should be understood as:

- OpenClaw loads a skill
- that skill teaches the agent to call the local `omicverse claw` CLI
- `omicverse claw` becomes a code-generation interface for OmicVerse workflows

This is different from treating OpenClaw as a generic shell wrapper. The correct mental model is: expose `omicverse claw` to OpenClaw as a local capability.

## 1. What OpenClaw Actually Provides

According to the OpenClaw docs:

- skills live in `<workspace>/skills` or `~/.openclaw/skills`
- a skill is a directory with a `SKILL.md`
- user-invocable skills can be called with `/skill <name> ...`
- skills can require binaries on `PATH`
- the agent uses the `exec` tool to run local shell commands

So the OmicVerse integration point is not a custom OpenClaw protocol. It is a skill that instructs OpenClaw to run the OmicVerse CLI.

## 2. Integration Goal

The goal is to give OpenClaw a stable interface like:

```bash
omicverse claw "basic qc and clustering"
```

or, when you want a deterministic output file:

```bash
omicverse claw --output omicverse_generated.py "basic qc and clustering"
```

In other words, OpenClaw should treat `omicverse claw` as the local backend that turns a natural-language request into OmicVerse code.

## 3. Recommended Architecture

The most practical architecture is:

1. Install `omicverse` on the same machine where OpenClaw can run `exec`
2. Create an OpenClaw skill such as `omicverse_claw`
3. In that skill, instruct OpenClaw to call the local `omicverse claw` command
4. Let OpenClaw return or save the generated Python code

This keeps responsibilities clean:

- OpenClaw handles conversation and skill routing
- `omicverse claw` handles OmicVerse code generation
- code execution remains a separate downstream decision

## 4. Create a Workspace Skill

Create the skill directory in one of OpenClaw's supported locations:

- workspace skill: `<workspace>/skills/omicverse-claw`
- user skill: `~/.openclaw/skills/omicverse-claw`

For a project-local integration, the workspace path is usually the best choice.

Example:

```bash
mkdir -p ./skills/omicverse-claw
```

Then create `./skills/omicverse-claw/SKILL.md` with content like this:

```md
---
name: omicverse_claw
description: Generate OmicVerse analysis code by calling the local omicverse claw CLI.
user-invocable: true
metadata: {"openclaw":{"requires":{"bins":["omicverse"]}}}
---

# OmicVerse Claw

Use this skill when the user wants OmicVerse analysis code, pipelines, or function-based workflow snippets.

Treat the user input after `/skill omicverse_claw` as the exact natural-language prompt for OmicVerse code generation.

Always call the local OmicVerse CLI with OpenClaw's `exec` tool instead of inventing the workflow directly.

Preferred command:

`omicverse claw --output "{baseDir}/omicverse_claw_latest.py" "<user request>"`

Rules:

- Preserve the user's request faithfully.
- Prefer `--output` so the generated code is saved deterministically.
- If the user asks for debugging or registry details, add `--debug-registry`.
- After the command succeeds, read `{baseDir}/omicverse_claw_latest.py` and return the generated Python code.
- If the command fails, report stderr and explain whether the `omicverse` binary is missing from `PATH`.
```

This gives OpenClaw a dedicated OmicVerse code-generation skill.

Recommended directory layout:

```text
your-openclaw-workspace/
├── skills/
│   └── omicverse-claw/
│       └── SKILL.md
└── ...
```

## 5. How to Call It from OpenClaw

After OpenClaw refreshes skills, you can call it in either of these ways.

Explicit skill invocation:

```text
/skill omicverse_claw basic qc and clustering
```

Another example:

```text
/skill omicverse_claw generate code for harmony batch correction and leiden clustering
```

If your OpenClaw surface exposes native skill commands, the skill may also be available directly as:

```text
/omicverse_claw basic qc and clustering
```

`/skill <name> ...` is the safest documented form.

If the skill does not appear immediately, start a new OpenClaw session. OpenClaw snapshots eligible skills at session start and reuses that list across turns.

## 6. What Command OpenClaw Should Run

For one-off generation, the skill should usually run:

```bash
omicverse claw --output "{baseDir}/omicverse_claw_latest.py" "<user request>"
```

For debugging:

```bash
omicverse claw --debug-registry --output "{baseDir}/omicverse_claw_latest.py" "<user request>"
```

For repeated use on the same machine, the skill can prefer the daemon-backed path:

```bash
omicverse claw --use-daemon --output "{baseDir}/omicverse_claw_latest.py" "<user request>"
```

That is the real interface OpenClaw should rely on.

## 7. Host vs Sandbox

OpenClaw's `exec` tool can run in a sandbox or on the gateway host.

For OmicVerse, the usual cases are:

- if `omicverse` is installed on the OpenClaw host, run on the host
- if your OpenClaw agent is sandboxed, either install `omicverse` inside the sandbox or route execution to the host

If OpenClaw cannot find the binary, check:

- `omicverse` is on `PATH`
- the skill gate `requires.bins` matches the actual install
- your `exec` host/security settings allow the command to run

## 8. Plugin Packaging

If you want to distribute this integration in OpenClaw's plugin style, package the same skill inside a plugin repo under a `skills/.../SKILL.md` directory.

OpenClaw plugins can ship skills, and those skills load when the plugin is enabled. In that model:

- the plugin provides the skill
- the skill teaches the agent to call `omicverse claw`
- OmicVerse remains the actual local CLI backend

For many users, a workspace skill is enough. A plugin is only needed when you want to distribute or version the integration.

A practical split is:

- use a workspace skill when you only need the integration in one repo
- use `~/.openclaw/skills` when you want it available across projects
- use an OpenClaw plugin when you want a distributable package with a manifest and config

## 9. Recommended User Prompts

These work well as OpenClaw skill inputs:

```text
basic qc and clustering
generate code for PCA, neighbors, UMAP, and leiden
find marker genes for each leiden cluster
write a minimal scRNA-seq annotation workflow with OmicVerse
prepare a harmony-based batch correction workflow
```

## 10. Important Notes

- `omicverse claw` is a code-generation interface, not a chat runtime
- OpenClaw should use it to generate OmicVerse code, not replace it with ad hoc shell logic
- `stdout` is the code payload; `stderr` is for init logs and debug information
- `--output` is the most stable pattern for OpenClaw skill usage
- daemon mode is useful only when OpenClaw will call `omicverse claw` repeatedly

## 11. Related Pages

- Full CLI tutorial: [OmicVerse Claw CLI](t_claw_cli.md)
- Jarvis overview: [Msg Bot Overview](t_msg_bot_overview.md)
