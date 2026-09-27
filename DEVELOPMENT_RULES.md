# Meta Voice Bot - Development Rules

These are the strict rules for developing this project.

1. **Local-First Deployment Protocol**
   - *Rule*: Never push unverified code to production.
   - *Details*: Whenever adding a new feature, editing existing code, or performing an upgrade, you MUST test and verify the changes in a local environment first (e.g., using a local uvicorn server). Code must ONLY be pushed to GitHub (and thus deployed to Render.com) when it has reached its absolute final, perfectly tested state.

2. **Secrets Management**
   - *Rule*: Never hardcode API keys or tokens.
   - *Details*: Always use .env files for local development and Render Environment Variables for production.

