import asyncio

from cybersec_agents import NetworkSecurityAgent
from cybersec_agents.utils.credentials import CredentialManager


async def main():
    # Initialize credential manager
    CredentialManager()

    # Create agents with different providers
    camel_agent = NetworkSecurityAgent(provider="camel")
    anthropic_agent = NetworkSecurityAgent(provider="anthropic")
    google_agent = NetworkSecurityAgent(provider="google")

    # Sample network data
    network_data = {
        "traffic_type": "http",
        "packet_count": 1000,
        "suspicious_ips": ["192.168.1.100"],
    }

    # Get analysis from multiple providers
    results = await asyncio.gather(
        camel_agent.analyze_traffic(network_data),
        anthropic_agent.analyze_traffic(network_data),
        google_agent.analyze_traffic(network_data),
    )

    return {
        "unified_threats": list(
            {threat for result in results for threat in result["threats"]}
        ),
        "provider_results": results,
    }


if __name__ == "__main__":
    asyncio.run(main())
