### 004. Server Hosting Preference: Render for now, Koyeb later

**LEARNING SOURCE:** Conversation on 2026-09-28 regarding free tier hosting alternatives (Render vs Koyeb).
**BUG/MISTAKE RECORDED:** The agent attempted to migrate or suggest migrating the codebase to other platforms due to Render's free tier sleep mode.
**PERMANENT RULE:** The project currently uses Render with an internal self-ping (cron) task to stay awake. Keep this setup running. The user explicitly requested to save this plan: in the future, when we want a "full setup", we will migrate to Koyeb to utilize its 24/7 free tier without sleep limitations. Do not force a migration right now.
