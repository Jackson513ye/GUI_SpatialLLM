# Spatial LLM
Current LLM models lack deep spatial understanding and the ability to reason over geolocation and vision inputs. For complex spatial queries users need manual interpretation. This project aims to bridge the gap by enabling a chatbot to understand user queries and extract relevant spatial and visual data automatically.

# GitRules
### Goals
- Keep main always deployable
- Work in short branches (not more than a few days). All changes go via Pull Request (PR)!
- Segher is the sole reviewer and merger
- Small reviewable commits, in clean code using comments and error handling

### Daily flow 
- Sync: git switch main && git pull --ff-only
- Branch: git switch -c feature/<short-task-name>
- Code → Commit (small, focused): git add -p && git commit -m "feat: ..."
- Push & PR: git push -u origin HEAD
- Wait for Segher to update/rebase and Squash & Merge when green
- Clean up locally: git switch main && git pull --ff-only && git branch -d <branch>

### Rules
- PR use so main stays protected
- use prefix feature/<name>, fix/<name>,  chore/<name>
