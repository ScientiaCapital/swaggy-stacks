# Security Guidelines

## API Key Management

### ⚠️ CRITICAL: Never Commit API Keys to Git

This repository uses multiple external services that require API keys. **Never commit actual API keys to version control.**

### Exposed API Keys - Immediate Action Required

If you've accidentally committed API keys:

1. **Rotate the keys immediately** in the respective service dashboards:
   - [Alpaca Markets](https://alpaca.markets/) - Trading API keys
   - [Anthropic Console](https://console.anthropic.com/) - Claude API keys
   - [OpenAI Platform](https://platform.openai.com/) - GPT API keys
   - [Perplexity AI](https://www.perplexity.ai/) - Research API keys

2. **Remove from git history** (if needed):
   ```bash
   # Use git-filter-repo (recommended) or BFG Repo-Cleaner
   # Contact your team lead before rewriting history on shared branches
   ```

3. **Update local environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your new API keys
   ```

### Secure Configuration

#### Environment Variables (.env)

All sensitive configuration should be stored in `.env` files:

```bash
# Copy the example file
cp .env.example .env

# Edit with your actual keys
vim .env  # or your preferred editor
```

**The `.env` file is excluded from git via `.gitignore`**

#### Required API Keys

1. **Alpaca Trading API** (Required)
   - Get keys: https://alpaca.markets/
   - Use paper trading keys for development
   - Set in `.env`:
     ```
     ALPACA_API_KEY=your_key_here
     ALPACA_SECRET_KEY=your_secret_here
     ALPACA_BASE_URL=https://paper-api.alpaca.markets
     ```

2. **Anthropic API** (Optional - for AI features)
   - Get keys: https://console.anthropic.com/
   - Set in `.env`:
     ```
     ANTHROPIC_API_KEY=sk-ant-...
     ```

3. **Other AI APIs** (Optional)
   - OpenAI: https://platform.openai.com/
   - Perplexity: https://www.perplexity.ai/
   - Google AI: https://ai.google.dev/

#### Kubernetes Secrets

For production deployments:

```bash
# Never commit actual secrets to k8s/base/secrets.yaml
# Use sealed-secrets or external secret management:

# Option 1: Sealed Secrets
kubectl apply -f k8s/base/sealed-secrets.yaml

# Option 2: External Secrets Operator
kubectl apply -f k8s/base/external-secrets.yaml

# Option 3: Manual secrets (development only)
kubectl create secret generic swaggy-stacks-secrets \
  --from-literal=ALPACA_API_KEY=your_key \
  --from-literal=ALPACA_SECRET_KEY=your_secret \
  --namespace=swaggy-stacks
```

### Files That Should Never Contain Real Keys

- `runpod-setup.sh` - Uses placeholder values only
- `.env.example` - Template file with examples
- `k8s/base/secrets.yaml` - Base64 placeholders only
- `CLAUDE.md` - Documentation only
- Any files committed to git

### Files Excluded from Git (.gitignore)

The following are automatically excluded:

```
.env
.env.local
.env.*.local
secrets/
*.pem
*.key
*.crt
```

### Security Checklist

Before committing code:

- [ ] No API keys in source code
- [ ] No passwords in configuration files
- [ ] `.env` file not tracked by git
- [ ] All secrets use placeholder values in examples
- [ ] Kubernetes secrets use external secret management
- [ ] RunPod/setup scripts use placeholders only

### Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. Email the maintainers directly
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Best Practices

1. **Use different keys for different environments**
   - Development: Paper trading keys
   - Staging: Separate paper trading keys
   - Production: Real trading keys (with appropriate risk limits)

2. **Rotate keys regularly**
   - At least every 90 days
   - Immediately after any suspected compromise
   - When team members leave

3. **Limit key permissions**
   - Use read-only keys when possible
   - Set trading limits on Alpaca keys
   - Use separate keys for different services

4. **Monitor key usage**
   - Check API logs regularly
   - Set up alerts for unusual activity
   - Track which services use which keys

### Incident Response

If keys are compromised:

1. **Immediate** (within 5 minutes):
   - Rotate all affected keys
   - Review recent trading activity
   - Check for unauthorized API access

2. **Short-term** (within 1 hour):
   - Audit git history for key exposure
   - Review all recent commits
   - Check deployment logs

3. **Follow-up** (within 24 hours):
   - Document the incident
   - Update security procedures
   - Train team on prevention
   - Consider additional security measures

## Additional Resources

- [Alpaca API Security](https://alpaca.markets/docs/api-documentation/api-v2/security/)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [OWASP API Security](https://owasp.org/www-project-api-security/)
