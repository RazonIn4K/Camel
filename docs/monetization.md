# Monetization Guide

## API Access Control
1. Configure API keys in config.yaml:
```yaml
monetization:
  api_keys:
    enabled: true
    tiers:
      basic: 1000
      premium: 5000
      enterprise: unlimited
```

## Usage Tracking
```python
from cybersec_agents import CyberSecurityService

service = CyberSecurityService(
    enable_monetization=True,
    billing_id="customer123"
)

# Usage tracking
usage_stats = service.get_usage_statistics()
print(f"Tokens used: {usage_stats['tokens']}")
print(f"Cost: ${usage_stats['cost']}")
```

## Payment Integration
1. Stripe setup in monetization_setup.py:
```python
import stripe
stripe.api_key = "sk_test_..."

def process_subscription(customer_id, tier):
    return stripe.Subscription.create(
        customer=customer_id,
        items=[{"price": tier_prices[tier]}]
    )
```