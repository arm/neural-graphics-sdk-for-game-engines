<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: MIT -->

# Agent workflows

Published SDK-facing skills live under `.agents/skills/<name>/SKILL.md`.
Agent clients that support this layout can discover them; otherwise open the
skill explicitly in your agent client.

Available workflows:

| Skill | Purpose |
|---|---|
| [sdk-build](skills/sdk-build/SKILL.md) | Configure/build supported SDK targets, samples, and unit tests using existing commands. |
| [sdk-samples](skills/sdk-samples/SKILL.md) | Run bounded NSS/NFRU samples and diagnose runtime, layer, and asset failures. |
| [sdk-validation](skills/sdk-validation/SKILL.md) | Select focused unit tests, interpret diagnostics, and check a supplied SDK bundle. |
| [dataset-validation](skills/dataset-validation/SKILL.md) | Perform static NSS/NFRU specification/profile conformance checks and write Markdown/JSON reports; no replay or pixel-domain diagnostics. |
| [jitter-validation](skills/jitter-validation/SKILL.md) | Analyze NSS jitter sequence and pixel-domain compensation with the standard PNG/score tool, compare debug-view replays, and write a Markdown report. |
| [nss-integration-debug](skills/nss-integration-debug/SKILL.md) | Orchestrate dataset and jitter validation, diagnose exposure/histogram and motion vectors, and aggregate results in Markdown. |

Example request:

> Explore this source tree for Agent workflow instructions relevant to the
> current task. Use the best matching workflow and follow its documented
> commands and safety constraints.

For direct invocation without an agent, use the
[build script](../build.py), [sample guide](../samples/README.md),
[unit-test guide](../unit_test/README.md), or
[validator command](../tools/dataset_validator/README.md). NSS captures can
also be inspected with the
[offline integration diagnostics](../tools/nss_integration_debug/README.md).

Each skill guides the agent through existing repository commands. Select the
relevant skill explicitly if the client does not discover it. Resolve the SDK
root relative to the selected skill and use the host's native shell/path syntax;
a published source tree need not have `.git`.

By default, generated skill reports and diagnostic artifacts are written under
the repository-root `agents_output/` directory. This directory is ignored by
Git and is not part of SDK publication.

## Repository guidance discovery

Before implementing a feature, inspect the checkout for repository- and
directory-level coding, contribution, and task guidance. Use the agent
client's normal instruction-discovery mechanism when available; otherwise
search from the repository root toward the files being changed. Read the
applicable guidance before editing and prefer the guidance most specific to
those files. Do not assume repository-only guidance is present in every
published source package.

## Publication and contribution

Published skills must work with repository-supported commands and public
dependencies. Do not put private paths, credentials, partner-specific
instructions, or unpublished implementation details in them. Keep run reports
outside the skill tree.

Use lowercase kebab-case names and
`.agents/skills/<name>/SKILL.md` with a matching frontmatter `name`. Each skill
needs a task-specific `description`, required tools, inputs, existing commands,
outputs, and failure handling. Treat the description as a concise task-routing
summary, not a second copy of SDK support matrices, command options, API
contracts, or implementation behavior. In the skill body, link to the
authoritative SDK guide or tool README for those details and follow that source
when behavior or options change. Keep only skill-specific orchestration, safety
constraints, decision-making, and evidence/reporting instructions that the
linked documentation does not own. If no authoritative documentation exists
for a reusable SDK behavior, add or update it in the SDK documentation and link
it from the skill rather than maintaining parallel descriptions.

Adding a file under `.agents/skills` does not approve it for publication. During
review, confirm the command exists, document inputs/outputs and error handling,
review all referenced files for public suitability, and exercise the workflow.
Then add each approved file individually to the repository's release whitelist.
Never whitelist an entire agent directory.

For each change, record representative commands, effective configuration,
process results, artifact/log locations, and unavailable platforms outside this
public tree. Exercise failure handling as well as the successful path; a sample
exit code alone is insufficient. The publication review must compare generated
agent files with the approved exact-file list. Recheck links against the
published tree, not only the engineering checkout.
