# MetalSplat2 — multi-agent orchestrator

## Project context
This is the working directory for two coordinated agent pipelines:

1. **PLY edit pipeline** — takes a `.ply` Gaussian splat and a natural language prompt, edits it in place (no reconstruction), renders the result, and verifies it visually
2. **Verification pipeline** — builds, renders, and visually scores splat output against a rubric and reference images

The VisionOS app layer is out of scope here. We operate purely on the visualization layer.

## Paths
- Working dir: `/Users/gvwert/Development/multi_agent`
- PLY input: `/Users/gvwert/Development/multi_agent/input/current.ply`
- PLY output: `/Users/gvwert/Development/multi_agent/output/edited.ply`
- Assets (injection source PLYs): `/Users/gvwert/Development/multi_agent/assets/`
- Segments cache: `/Users/gvwert/Development/multi_agent/.claude/segments/`
- Edit specs: `/Users/gvwert/Development/multi_agent/.claude/edits/`
- Backups: `/Users/gvwert/Development/multi_agent/.claude/backups/`
- Golden screenshots: `/Users/gvwert/Development/multi_agent/.claude/golden/`
- Reference images: `/Users/gvwert/Development/multi_agent/.claude/reference/`
- Source project: `/Users/gvwert/Development/xcode/MetalSplat2`

## Render server
- Base URL: `http://localhost:FILL_IN_PORT`
- Always confirm server is running before dispatching any agent that needs renders

## Available agents
| Agent | Role |
|---|---|
| `segmenter` | renders views, maps surfaces → Gaussian index lists |
| `edit-planner` | parses edit prompt → structured delta spec |
| `ply-editor` | executes delta spec → writes edited PLY |
| `builder` | modifies splat generation code |
| `compile-checker` | runs xcodebuild, returns pass/fail + errors |
| `visual-verifier` | renders, screenshots, scores against rubric |
| `regression` | stores golden screenshots, diffs for drift |

## Pipeline A — PLY edit pipeline (primary)
Use when the user provides an edit prompt against an existing PLY.

**Stage order (always sequential):**
1. `segmenter` — identify target region
2. `edit-planner` — decide strategy + generate delta spec
3. `ply-editor` — apply edits to PLY
4. `visual-verifier` — render edited PLY and score it
5. `regression` — update golden on pass

**Retry logic:**
- `visual-verifier` fail → send critique to `edit-planner`, regenerate spec, re-run `ply-editor`, max 3 retries
- On max retries: surface last failure and critique to user, do not loop further
- `ply-editor` fail → restore from backup, report to user

**What counts as done:**
- `visual-verifier` returns weighted score ≥ 7.0 with no dimension below 5/10
- `regression` confirms golden updated

## Pipeline B — code build + verify pipeline
Use when modifying splat generation code (not editing an existing PLY).

**Stage order:**
1. `builder` — implement the feature
2. `compile-checker` — verify it builds
3. `visual-verifier` — render and score
4. `regression` — update golden on pass

**Retry logic:**
- `compile-checker` fail → send stderr to `builder`, max 3 retries
- `visual-verifier` fail → send critique to `builder`, max 3 retries

## Routing rules

**Use Pipeline A when:**
- User says "change X to Y", "make X look like Y", "remove X", "add X to the scene"
- A `.ply` file already exists in `input/current.ply`

**Use Pipeline B when:**
- User says "implement feature X", "fix the generation code", "add support for Y in the pipeline"
- The task involves Swift/Python/shader code changes

**Run in main session (no agent delegation) when:**
- User is planning, asking questions, or reviewing results
- A pipeline hit max retries and needs human input
- The edit prompt is ambiguous between strategies

## Sub-agent routing rules
- Never parallelize pipeline stages — each depends on the previous output
- Never run `visual-verifier` if the render server is not responding
- Never run `regression` on a failed verification
- Never run `ply-editor` without a valid delta spec from `edit-planner`
- Strategy C2 (reconstruction injection) must be flagged to user before executing — it is slow

## Directory initialization
On first run, ensure these dirs exist:
```bash
mkdir -p /Users/gvwert/Development/multi_agent/{input,output,assets}
mkdir -p /Users/gvwert/Development/multi_agent/.claude/{segments/renders,edits,backups,golden,reference}
```
