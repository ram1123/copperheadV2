---
name: coordinate
description: Coordinate specialized agents for complex CMS analysis tasks while preserving project memory.
disable-model-invocation: true
---

# Agent Coordination

This skill is executed by the main Claude session.

## 1. Define the task

Identify:

- requested outcome;
- relevant files and components;
- applicable era and campaign;
- constraints;
- acceptance criteria;
- whether file modification is authorized.

Do not delegate a task merely because agents are available.

## 2. Retrieve relevant memory

Read `.claude/reports/registry.md`.

Open only reports relevant to the current task. Do not load all reports.

## 3. Select specialists

Use specialized agents only when they contribute distinct expertise.

Typical assignments:

- `physics-reviewer`: object definitions, selections, corrections,
  systematic uncertainties, and physics interpretation;
- `code-reviewer`: implementation bugs, masks, weights, indexing,
  configuration, and numerical problems;
- `test-specialist`: reproducible tests, validation, and failure isolation.

## 4. Prepare bounded tasks

Every delegated task must state:

- exact question or deliverable;
- relevant context;
- files or directories to inspect;
- applicable era and dataset;
- constraints;
- whether editing is allowed;
- expected response format.

Sub-agents must not read the registry, invoke this skill, or expand the scope
of their assignment.

## 5. Choose execution order

Run tasks in parallel when they are independent.

Run tasks sequentially when:

- one result determines the next task;
- several agents would edit the same files;
- implementation depends on an unresolved physics decision.

## 6. Synthesize results

The main agent must:

- compare findings;
- resolve or report disagreements;
- reject unsupported claims;
- eliminate duplicate findings;
- identify missing evidence;
- make the final recommendation.

Sub-agent output is evidence, not the final decision.

## 7. Verify

Check the actual repository state.

When applicable:

- inspect changed files;
- inspect `git diff`;
- run tests;
- verify output files;
- check that the implementation matches the requested physics behavior.

Do not rely only on an agent's statement that work succeeded.

## 8. Preserve useful memory

For substantial work, create one concise report under:

- `.claude/reports/decisions/`;
- `.claude/reports/investigations/`;
- `.claude/reports/implementations/`.

Add one corresponding entry to `.claude/reports/registry.md`.

Record durable conclusions, decisions, validation results, and unresolved
questions. Do not preserve the entire conversation.