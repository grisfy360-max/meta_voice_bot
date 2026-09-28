### 005. Configure and Document Facebook API Permissions Upfront

**LEARNING SOURCE:** User instruction on 2026-09-28 during Phase 2 planning (Auto-comment & Image Recognition).
**BUG/MISTAKE RECORDED:** Adding new features like "auto-comment" required mid-development changes to Facebook Developer App permissions (e.g., pages_manage_engagement) and Webhook subscriptions (eed), which interrupts the workflow.
**PERMANENT RULE:** All required Facebook App permissions (e.g., pages_messaging, pages_manage_engagement, whatsapp_business_messaging) and Webhook events (messages, eed) MUST be documented and configured at the very beginning of the project/client setup. Update the project's documentation/README so clients or agents configure them proactively before writing the integration code.
