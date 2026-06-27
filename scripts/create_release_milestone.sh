#!/usr/bin/env bash
# Create (or reuse) a GitHub release milestone and optionally assign issues/PRs.
#
# Usage:
#   ./scripts/create_release_milestone.sh 0.2.0
#   ./scripts/create_release_milestone.sh 0.2.0 954 944 930 952 953
#
# Requires: gh CLI authenticated with issues:write on Dooders/AgentFarm.

set -euo pipefail

REPO="${AGENTFARM_REPO:-Dooders/AgentFarm}"
VERSION="${1:?Usage: $0 <version> [issue numbers...]}"
shift || true

if ! echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'; then
  echo "Version must look like 0.2.0 or 0.2.0-rc1 (got: $VERSION)" >&2
  exit 1
fi

SCOPE_FILE="docs/milestones/${VERSION}.md"
if [ -f "$SCOPE_FILE" ]; then
  DESCRIPTION="Release ${VERSION}. Scope: docs/milestones/${VERSION}.md · Process: docs/RELEASE.md"
else
  DESCRIPTION="Release ${VERSION}. See docs/RELEASE.md for the release process."
fi

EXISTING=$(gh api "repos/${REPO}/milestones" --paginate --jq ".[] | select(.title==\"${VERSION}\") | .number" | head -n 1)
if [ -n "$EXISTING" ]; then
  echo "Milestone ${VERSION} already exists: https://github.com/${REPO}/milestone/${EXISTING}"
  MILESTONE_NUMBER="$EXISTING"
else
  MILESTONE_NUMBER=$(gh api "repos/${REPO}/milestones" \
    -f title="${VERSION}" \
    -f state="open" \
    -f description="${DESCRIPTION}" \
    --jq '.number')
  echo "Created milestone ${VERSION}: https://github.com/${REPO}/milestone/${MILESTONE_NUMBER}"
fi

for NUMBER in "$@"; do
  gh api "repos/${REPO}/issues/${NUMBER}" -X PATCH -f milestone="${MILESTONE_NUMBER}" >/dev/null
  echo "Assigned #${NUMBER} to milestone ${VERSION}"
done
