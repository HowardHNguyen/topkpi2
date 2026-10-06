# Git & CI/CD guide for TopKPI2

## Mental model
- **Commit** = a saved snapshot with a message. History is a chain of commits.
- **Branch** = a separate line of work. `main` is the live, stable line.
- **Pull Request (PR)** = a GitHub request to merge a branch into `main`. CI runs on it.
- **Tag** = a permanent label on a commit, used for versions (`v1.0.0`).
- **Local** (your Mac) vs **remote** (`origin` = GitHub). `push` sends, `pull` receives.

## Every change, step by step
```bash
git switch main && git pull                 # 1. start from latest main
git switch -c fix/short-description         # 2. new branch
# ...edit files...
git status                                  # 3. what changed?
git diff                                    #    see the exact edits
git add app.py                              # 4. stage chosen files
git commit -m "Fix: explain what and why"   # 5. save snapshot
git push -u origin fix/short-description    # 6. send to GitHub
gh pr create --fill                         # 7. open PR (or use the GitHub button)
```
8. Wait for the green check from CI on the PR, review "Files changed", click **Merge**.
9. Back locally: `git switch main && git pull && git branch -d fix/short-description`

## Releasing a version
1. Add a section to `CHANGELOG.md`.
2. After merging to `main`:
```bash
git switch main && git pull
git tag -a v1.1.0 -m "Short summary"
git push origin v1.1.0
```
Version numbers: `MAJOR.MINOR.PATCH` - PATCH = bug fix, MINOR = new feature, MAJOR = breaking change.

## Undo cheatsheet
| Goal | Command |
|---|---|
| Throw away uncommitted edits to a file | `git restore app.py` |
| Unstage a file | `git restore --staged app.py` |
| Undo a commit already pushed (safe) | `git revert <hash>` |
| Browse history | `git log --oneline --graph` |
| Compare to a release | `git diff v1.0.0 -- app.py` |
| Return the code to a release (read-only look) | `git switch --detach v1.0.0` then `git switch main` |
| Park unfinished work | `git stash` / `git stash pop` |

Avoid `git reset --hard` and `git push --force`; they can destroy work.

## CI/CD here
- **CI** (`.github/workflows/ci.yml`): on every PR and push to `main`, GitHub installs
  dependencies, compiles `app.py` and runs `tests/` (every page must render).
  Run it locally: `pip install -r requirements-dev.txt && pytest -q`
- **CD**: if the app is hosted on Streamlit Community Cloud from `main`, it redeploys
  automatically on each merge. So: merge only green PRs.
- Recommended: GitHub > Settings > Branches > add a rule for `main` requiring a PR and
  the `test` check to pass before merging.
