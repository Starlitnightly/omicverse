# Security Policy

## Supported Versions
Use this section to tell people which versions of **omicverse** are currently receiving security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.7.x   | :white_check_mark: |
| 1.6.x   | :white_check_mark: |
| < 1.6   | :x:                |

> **Note:** Python **3.9** and earlier branches are no longer maintained.  
> Supported Python versions: **3.10**, **3.11** :contentReference[oaicite:2]{index=2}.

## Reporting a Vulnerability
If you discover a security vulnerability in **omicverse**, please follow these steps :contentReference[oaicite:3]{index=3}:

1. **Open an issue** on GitHub at  
   `https://github.com/Starlitnightly/omicverse/issues/new?labels=security`  
2. Tag the issue with **security** and provide a clear, concise description, including:  
   - Component or version affected  
   - Steps to reproduce  
   - Any proof-of-concept code, if available  
3. We aim to **acknowledge** all reports within **72 hours** and provide regular status updates :contentReference[oaicite:4]{index=4}.  
4. If the vulnerability is accepted, it will be fixed in the next patch and you will be **credited** in the release notes. If declined, we will explain our rationale.

## Security Response Process
We follow industry-standard triage and remediation practices:

1. **Triage & Acknowledgment**  
   - Upon report, a security maintainer assigns a severity (Critical / High / Medium / Low) based on CVSS guidelines.  
2. **Investigation & Remediation**  
   - Patches are developed in a private branch.  
   - Automated tests and static analysis (e.g., CodeQL) are run to prevent regressions :contentReference[oaicite:5]{index=5}.  
3. **Disclosure & Release**  
   - A coordinated disclosure date is set.  
   - A patched version is released, and a public advisory is posted.  
4. **Post-mortem & Hardening**  
   - We review root causes and update continuous integration checks (e.g., dependency scanning via Dependabot) :contentReference[oaicite:6]{index=6}.

## Secure Development Best Practices
We recommend contributors follow these guidelines to minimize risk:

- **OWASP Top 10 Awareness**: Be mindful of Injection, Broken Access Control, Cryptographic Failures, etc. :contentReference[oaicite:7]{index=7}.  
- **Secret Management**: Never commit credentials or API keys to the repository; use environment variables or vault services :contentReference[oaicite:8]{index=8}.  
- **Dependency Hygiene**: Keep dependencies up to date and monitor for known vulnerabilities with Dependabot alerts and CodeQL scanning :contentReference[oaicite:9]{index=9}.  
- **Branch Protection**: Require pull-request reviews, status checks, and signed commits for the `main` branch :contentReference[oaicite:10]{index=10}.  

## Contact
For any security questions, please reach out to the security team:

- Email: **starlitnightly@gmail.com**  
- Alternative: Open a GitHub issue with the **security** label.

---

_Last updated: 2025-05-24_  
