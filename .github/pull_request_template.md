## Summary

<!--
Explain what changed and why. Keep this focused on the reviewer-facing context,
not a file-by-file changelog.
-->

## Related Issue

<!--
Use "Fixes #123", "Closes #123", or "N/A" if there is no associated issue.
-->

## Type of Change

<!-- Delete entries that do not apply. -->

- Bug fix
- Feature
- Refactor
- Documentation
- Tests
- Build / CI / tooling
- Breaking change

## Affected Areas

<!-- Delete entries that do not apply and add notes where useful. -->

- Training loop / policy / loss
- Rollout engine or request/response flow
- Reward service or reward components
- Ray actors, placement, or distributed orchestration
- Hydra configs, experiment recipes, or launch scripts
- Model bundle, pipeline, stages, LoRA, or FSDP wrapping
- Data, checkpoints, or mounted artifacts
- Documentation only

## Behavior and Compatibility

### Current behavior

<!-- What behavior, limitation, or bug exists before this PR? -->

### New behavior

<!-- What changes after this PR? -->

### Compatibility impact

<!--
Mention config changes, checkpoint compatibility, data format changes, API changes,
GPU/resource requirement changes, migration steps, or write "N/A".
-->

## Test Plan

<!--
List the exact commands or jobs you ran. Include relevant model, recipe, hardware,
GPU count, and dataset/checkpoint details for training or rollout changes.
-->

- `SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure`
- `pytest`
- Hydra config validation, for example:
  `python -m unirl.train_diffusion --config-name=diffusion_rl/<recipe> --cfg job --resolve`
- Training / rollout smoke test
- Not run; reason:

## User-Facing Release Note

<!--
If this PR changes user-visible behavior, write a concise release note.
If not, leave NONE.
-->

```release-note
NONE
```

## Reviewer Notes

<!--
Call out risky areas, known limitations, follow-up work, or anything reviewers
should inspect first. Use "N/A" if there is nothing special.
-->

## Checklist

- [ ] I have performed a self-review of the changed code.
- [ ] I have added or updated tests for behavior changes, or explained why tests are not needed.
- [ ] I have updated documentation or configs where needed.
- [ ] I have not added large generated files, local outputs, credentials, datasets, or model checkpoints.
- [ ] Breaking changes, migrations, and user-facing behavior changes are documented above.
