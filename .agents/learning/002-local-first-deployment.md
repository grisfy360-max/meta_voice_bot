# Local-First Deployment Protocol

## Rule
**Never push unverified code to production.**
Whenever adding a new feature, editing existing code, or performing an upgrade, you MUST test and verify the changes in a local environment first (e.g., using a local uvicorn server). Code must ONLY be pushed to GitHub (and thus deployed to Render.com) when it has reached its absolute final, perfectly tested state.

## Rationale
The user explicitly requested: "jekono feautre add edit upgrade jai hok final porjon na gele eta amoni rakheb" (whatever feature is added, edited, or upgraded, until it reaches the final stage, keep it like this). Pushing untested code to Render causes unnecessary downtime, deployment delays (1-3 minutes), and exposes bugs to the live environment.

## Execution
1. Make code changes locally.
2. Run the application locally (e.g. uvicorn main:app --port 8000).
3. Verify the changes using local curl or HTTP requests.
4. If applicable, get user confirmation on the local state.
5. ONLY run git push when the user and agent agree the feature is 100% final.
