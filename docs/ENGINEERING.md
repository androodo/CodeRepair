# Engineering walkthrough

This document describes the decisions that matter to CodeRepair's correctness
and threat model. It intentionally avoids claims that are not backed by a run
artifact.

## Why a custom execution loop

The useful engineering surface is the loop itself: how provider messages become
privileged local actions, how failures return to the model, and how execution
stops. A framework would save little code here while hiding protocol and budget
behavior that should be auditable in an interview or security review.

`Agent.run` owns an append-only list of Messages API turns. It asks a deliberately
small `ModelClient` for one response at a time. Production uses
`AnthropicMessagesModel`; tests use `FakeModel`. This is not a generalized model
provider layer—the abstraction exists only to make the real loop deterministic
under test.

Current Anthropic API details matter. Tool input is already decoded JSON. Tool
results must follow the assistant tool-use turn immediately, and parallel calls
must receive results together. Current models may emit signed `thinking` blocks,
so CodeRepair serializes and replays the complete content rather than retaining
only visible text. The SDK performs bounded transient retries; CodeRepair does
not multiply them by default. Removed sampling parameters are not sent.

## Dispatch and error observations

The model sees strict JSON Schema definitions, but schema compliance is not a
security boundary. `ToolDispatcher` independently validates a small schema
subset, rejects unknown properties and tools, and closes over the workspace,
test command, image, and limits chosen before execution.

Calls in one response execute sequentially in block order. This preserves
deterministic semantics when one patch affects a later read. For every call ID,
the loop creates exactly one `tool_result`. Unknown tools, invalid arguments,
path denials, syntax failures, Docker failures, and internal exceptions become
bounded `is_error` observations rather than malformed history. The next model
turn can correct an ordinary failure without gaining additional authority.

## Repository isolation

The source worktree is treated as immutable input. Startup resolves the Git root,
records HEAD, and rejects any tracked, staged, or untracked change. It also
rejects bare repositories, submodules, and tracked links. `git archive HEAD`
produces a tracked snapshot without modifying `.git` and without copying history,
hooks, worktree secrets, or uncommitted data. Archive members are validated and
extracted into a private temporary directory, which receives its own baseline
commit solely for producing a clean unified diff.

A worktree or clone was not used because both copy more repository control state;
`git worktree` also mutates the original repository metadata. Rejecting dirty
input is less convenient than silently ignoring or including it, but makes the
patch's starting point explicit and preserves user work.

Every tool path passes one policy. It rejects NULs, native absolute/drive/UNC
forms, backslashes, empty and dot segments, parent traversal, case-insensitive
`.git`, sensitive filename patterns, non-regular targets, and every symlink
component. Repository instructions and code are untrusted context; none can
change this host-side enforcement.

## Atomic editing and syntax validation

Free-form shell patches are unnecessary authority. CodeRepair uses structured
exact-text operations. Each replacement must identify one occurrence, which
makes stale context and ambiguous broad edits fail visibly. The dispatcher
computes all final file bytes first, enforces protected/editable globs and size
limits, and parses every changed Python file. Only after all validation passes
are same-directory temporary files fsynced and atomically replaced. A rare
mid-commit failure triggers best-effort rollback from in-memory originals.

`ast`/compile validation proves only that Python can parse a file. It says
nothing about imports, types, runtime behavior, concurrency, security properties,
or whether the requested bug is fixed. That is why reports keep syntax, visible
tests, hidden tests, regressions, and verified success as separate fields.

## Restricted test execution

Only a trusted CLI or manifest value can select the pytest argv. The model-facing
`run_tests` schema is empty. Its handler cannot accept a command, Docker image,
mount, environment, or resource setting from model output.

The prebuilt container runs without network access, as UID/GID 10001, with all
capabilities dropped and `no-new-privileges`. It has a read-only root filesystem,
read-only source mount, bounded tmpfs, default seccomp, and CPU, memory, process,
file-descriptor, output, and host-time limits. The host home, Docker socket, API
key, SSH material, and cloud credentials are not mounted. A UUID name lets the
host kill and remove the exact container if the Docker client times out.

CodeRepair checks for the image locally and never pulls, installs repository
dependencies, enables a network, or falls back to host execution. Teams can build
a reviewed image for known dependencies. Containers reduce risk; kernel and
daemon vulnerabilities remain possible, so stronger isolation may be warranted
for actively adversarial code.

## Iteration and runaway prevention

Four independent limits stop the agent: successful model/tool cycles, dispatched
tool calls, model-reported tokens, and wall-clock duration. Each model request
also caps output tokens, and Docker has a separate test timeout. Exhaustion is an
execution status, not a successful repair. A visible pass is valid only for the
workspace generation that was tested; any later patch makes that result outdated.

The model may end its turn at any time. `end_turn` becomes
`completed_unverified` unless the latest generation has a visible test pass.
Refusal, context exhaustion, unexpected pause, and truncation are incomplete
states. This prevents natural-language confidence from being recorded as proof.

## Hidden-test isolation and evaluation

An evaluation task pins its start with either an exact Git commit or a canonical
source-tree SHA-256. A mismatch aborts before a model request. Each configuration
gets a fresh source snapshot. The agent workspace contains visible tests but no
hidden test filename, bytes, or mount.

After the loop ends, evaluation creates another fresh candidate copy and applies
only candidate source changes. Hidden target and regression suites are then
mounted read-only under `/evaluation` in a new container. Their results are not
sent back to the model. This separation catches repairs that overfit visible
tests and prevents the agent from editing independent checks.

Repair success means independent target tests pass. Visible pass rate remains a
separate diagnostic. A regression is behavior that passed in the untouched
baseline snapshot but fails after the candidate patch; infrastructure errors are
not silently relabeled as repair failures. Verified task success requires syntax,
visible tests, independent tests, and regression checks.

## What the baseline comparison measures

Both modes use the same configured model, task, start snapshot, repository tools,
and model budgets. The baseline does not receive `run_tests`; its candidate is
tested only after it stops. The iterative mode adds the one approved visible-test
tool and feeds those results into subsequent model turns.

This comparison asks whether in-loop visible feedback helps under the documented
configuration. It is not a perfect causal isolation: test calls consume wall
time, tool budget, and context, and the model can choose different trajectories.
Reports therefore include actual calls, iterations, latency, and tokens rather
than presenting only a success percentage.

The included fixtures are infrastructure checks. A credible public-task study
would pin 10–20 carefully reviewed bug commits, preserve licenses, provision
deterministic images, verify that hidden tests were not upstream-visible, repeat
stochastic runs, and disclose failures and budgets. CodeRepair does not claim
SWE-bench performance.

## Approval-gated Git publication

Repair generation and repository mutation are separate commands. `run` can only
write a temporary workspace and an external artifact directory. `apply` prints
the complete patch, checks the exact starting commit and clean status, and exits
without mutation unless `--approve` is present. It creates a new branch, applies
and stages the reviewed patch, verifies the staged path set, disables repository
hooks for the commit, and stops there by default.

Push and draft PR creation are separate flags. Draft PR creation uses `gh`, never
merges, and never pushes directly to the default branch. If a later operation
fails, CodeRepair leaves the new branch visible for inspection instead of
discarding data with an automatic hard reset.

## Maintainability tradeoffs

- Synchronous calls keep ordering and failure behavior easy to explain; parallel
  model sessions are an evaluator-level future optimization.
- Exact replacements are less expressive than arbitrary diffs but much easier to
  validate and roll back.
- A fixed pytest-only image excludes many repositories but avoids dependency
  installation as an agent-controlled code-execution channel.
- Clean-HEAD-only snapshots omit useful dirty work by design; explicit commits
  are a clearer reproducibility boundary.
- JSON/JSONL avoids a database and keeps every result portable and reviewable.
- One agent is enough. Multi-agent coordination would add cost and attribution
  complexity without solving a demonstrated v1 problem.

These constraints keep the important mechanisms small enough to audit line by
line while still exercising model protocols, secure process boundaries, Git,
evaluation design, and failure-aware backend engineering.
