# Security Policy

## Authorized-use notice

**injectkit is a defensive security tool.** It exists so that developers can
red-team their *own* LLM applications for prompt-injection weaknesses before an
attacker does — the same posture as "scan your own website" vulnerability
scanners.

By using injectkit you agree to run it **only** against:

- LLM endpoints, chatbots, agents, or models that **you own**, or
- targets you have **explicit, written authorization** to test.

Do **not** point injectkit at third-party services, public chatbots you do not
operate, or any system you are not authorized to assess. Doing so may violate
the target's terms of service and applicable law. The authors accept no
liability for misuse. See the [LICENSE](LICENSE) (MIT) for the full warranty
disclaimer.

Every report injectkit produces carries this authorized-use notice.

## Reporting a vulnerability in injectkit

If you discover a security vulnerability **in injectkit itself** (for example,
a way the tool could be coerced into attacking an unintended target, or a flaw
in how it handles credentials), please report it responsibly:

1. **Do not** open a public GitHub issue for the vulnerability.
2. Email the maintainer at the address on the project's GitHub profile, or open
   a private security advisory via GitHub's "Report a vulnerability" feature on
   the repository's **Security** tab.
3. Include a description, reproduction steps, and the impact you observed.

We aim to acknowledge reports within a few business days and to coordinate a
fix and disclosure timeline with you.

## Responsible disclosure of findings injectkit produces

Findings from scanning *your own* systems are yours to remediate. If injectkit
helps you find an injection issue in a **third-party** product you are
authorized to test, follow that vendor's responsible-disclosure process — do
not publish exploit details before they have had a reasonable chance to fix it.

## Handling secrets

- API keys (e.g. `ANTHROPIC_API_KEY`) are read from the environment and are
  never written to reports or committed to the repo.
- Do not paste real credentials into corpus YAML, config files, or issues.
