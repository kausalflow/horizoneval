# Use horizoneval from an AI assistant

horizoneval is built so that an AI assistant can run it and explain the answer, without computing a single number
itself. Every way in returns the same versioned JSON document (`result.json`); a test makes sure the Python API, the
run folder, the command line and the MCP server return byte-identical results.

## MCP server

```bash
pip install "horizoneval[mcp]"
horizoneval-mcp            # speaks MCP over stdio
```

Add it to an MCP client, for example Claude Desktop's `claude_desktop_config.json` or a project's `.mcp.json`:

```json
{"mcpServers": {"horizoneval": {"command": "horizoneval-mcp"}}}
```

Tools: `check` (pre-flight checks), `score` (evaluate and choose), `diff` (why two runs disagree), `read_result`,
`list_forecasters`. Each returns the result document as text and as structured content.

## Agent Skill

`skills/horizoneval/SKILL.md` in the repository teaches an assistant the workflow (check, score, explain) and how to
read the result in plain language. Copy the folder into your assistant's skills directory (for Claude Code:
`.claude/skills/`).

## Command line with JSON

```bash
horizoneval check cv.csv --train train.csv --json      # exit code 1 on an error
horizoneval score cv.csv --train train.csv --period 7 --json
horizoneval diff results/a results/b --json
```

## Function calling

```python
from horizoneval import tools

tools.definitions("anthropic")   # or "openai": tool definitions with JSON Schema inputs
tools.call("score", {"forecasts": "cv.csv", "train": "train.csv", "period": 7})
```

## Explaining a result

Two tools let an assistant explain a result to someone who is not a forecaster without inventing anything:

- `explain` (`horizoneval explain results/run`) returns the answer in plain sentences, how to read every table, the
  caveats that must be mentioned, why the numbers can be trusted, and `facts`: every number an explanation may quote,
  labelled in plain words, with its percentage, its "times smaller" ratio and its 95% interval. The same document
  is written to `explanation.json` in every run folder.
- `check_explanation` (`horizoneval explain results/run --check draft.txt`) lists every number in a draft that the
  result does not support, at the precision it is written with ("0.8" passes for 0.822, "37.4%" fails if nothing
  in the result rounds to it). It catches invented numbers; it cannot catch a true number attached to the wrong
  claim, which is why the facts carry plain labels.

The Agent Skill tells the assistant to explain only from `explain`, mention every warning, and run
`check_explanation` before it replies.

## The result document

Validated against `horizoneval/schema/result.schema.json` (`horizoneval.result.validate`). Its version,
`horizoneval_result`, changes when a field changes meaning or disappears. The main fields: `verdict` (plain
language), `choice` (per dataset: model, probability it is best, verified claim, winner's curse), `segments`,
`cost_disagreement`, `checks`, `failures`, `summary`, `leaderboard`, and `run` (settings and fingerprints). It holds
no timestamps or paths, so the same forecasts give the same document wherever they are scored.
