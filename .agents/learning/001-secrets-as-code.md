### 001. Enterprise Secrets as Code Strategy

**LEARNING SOURCE:** User architectural suggestion during Phase-1 webhook configuration.
**BUG/MISTAKE RECORDED:** The .env file logic wasn't fully scalable for large environments with multiple secrets.
**PERMANENT RULE:** For Phase 1, we use Render.com's manual Vault (Environment Variables) to store tokens like META_API_TOKEN because we have very few secrets. However, as the project scales to Phase 2+ and the number of secrets grows, we must migrate to a "Secrets as Code" architecture by encrypting the .env file (e.g., using SOPS, git-crypt, or Vault) before pushing to GitHub. This ensures deployments only require a single master decryption key.
