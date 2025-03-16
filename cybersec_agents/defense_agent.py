"""
DefenseAgent Module.

This module implements the DefenseAgent class, which is responsible for defending
systems against cyber threats. The agent analyzes input data for potential threats,
processes the information, and generates appropriate defensive responses.
"""

from typing import Any, Dict, Optional, List
import logging
from datetime import datetime, time
import re
from enum import Enum

class ThreatLevel(Enum):
    """Enumeration of possible threat levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class EventType(Enum):
    """Enumeration of possible event types."""
    LOGIN = "login"
    NETWORK = "network"
    SYSTEM = "system"
    SECURITY = "security"
    UNKNOWN = "unknown"

class DefenseAgent:
    """
    An AI agent responsible for defending a system against cyber threats.
    
    The DefenseAgent analyzes system data, network traffic, and security events
    to identify potential threats and generate appropriate defensive responses.
    It maintains a state of the system's security posture and can take actions
    to mitigate identified risks.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", log_level: str = "INFO"):
        """
        Initialize the DefenseAgent.

        Args:
            model_name: The name of the language model to use for analysis
            log_level: The logging level for the agent
        """
        self.model_name = model_name
        self.role = None
        self.input_data = None
        self.potential_threat = False
        self.threat_level = ThreatLevel.INFO
        self.last_analysis = None
        self.defensive_actions = []
        self.event_history = []
        self.business_hours = {
            "start": time(9, 0),  # 9 AM
            "end": time(17, 0)    # 5 PM
        }
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DefenseAgent initialized with model: {model_name}")

    def receive_input(self, input_data: Dict[str, Any]) -> None:
        """
        Receive data from the competition environment.

        Args:
            input_data: Dictionary containing system data, network traffic,
                       security events, or other relevant information.
        """
        self.logger.info("DefenseAgent receiving input data")
        self.input_data = input_data
        self.event_history.append({
            "timestamp": datetime.now().isoformat(),
            "data": input_data
        })
        self.logger.debug(f"Input data received: {input_data}")

    def process_input(self, input_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Process the received input and analyze for threats.

        Args:
            input_data: Optional data to process. If not provided, uses the
                       last received input data.
        """
        data = input_data or self.input_data
        if not data:
            self.logger.warning("No input data to process")
            return

        self.logger.info("DefenseAgent processing input for threats")
        
        # Analyze the input data for potential threats
        threat_indicators = self._analyze_threat_indicators(data)
        self.potential_threat = threat_indicators["has_threat"]
        self.threat_level = threat_indicators["threat_level"]
        
        # Store analysis results
        self.last_analysis = {
            "timestamp": datetime.now().isoformat(),
            "threat_indicators": threat_indicators,
            "threat_level": self.threat_level.value,
            "event_type": threat_indicators.get("event_type", EventType.UNKNOWN.value)
        }
        
        self.logger.info(f"Analysis complete. Threat level: {self.threat_level.value}")

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a response based on the threat analysis.

        Returns:
            Dictionary containing the recommended defensive actions and
            supporting information.
        """
        self.logger.info("DefenseAgent generating defensive response")
        
        if not self.last_analysis:
            self.logger.warning("No analysis available to generate response")
            return {"action": "continue_monitoring", "reason": "No analysis data"}

        # Generate appropriate defensive actions based on threat level
        response = self._generate_defensive_actions()
        self.defensive_actions.append({
            "timestamp": datetime.now().isoformat(),
            "action": response,
            "threat_level": self.threat_level.value
        })
        
        self.logger.info(f"Generated response for threat level: {self.threat_level.value}")
        return response

    def send_output(self) -> Dict[str, Any]:
        """
        Send the generated response to the competition environment.

        Returns:
            Dictionary containing the defensive actions and supporting
            information to be sent to the competition.
        """
        output = self.generate_response()
        self.logger.info("DefenseAgent sending output")
        self.logger.debug(f"Output being sent: {output}")
        return output

    def give_role(self, role: str) -> None:
        """
        Set the agent's role and responsibilities.

        Args:
            role: String describing the agent's role and responsibilities
        """
        self.role = role
        self.logger.info(f"DefenseAgent role set to: {role}")

    def _analyze_threat_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze input data for potential threat indicators.

        Args:
            data: Dictionary containing system data to analyze

        Returns:
            Dictionary containing threat analysis results
        """
        if not isinstance(data, dict):
            return {
                "has_threat": False,
                "threat_level": ThreatLevel.INFO,
                "timestamp": datetime.now().isoformat()
            }

        # Define threat patterns and their associated threat levels
        threat_patterns = {
            ThreatLevel.CRITICAL: [
                r"critical.*vulnerability",
                r"remote.*code.*execution",
                r"privilege.*escalation",
                r"data.*breach",
                r"ransomware",
                r"zero.*day"
            ],
            ThreatLevel.HIGH: [
                r"malware",
                r"exploit",
                r"injection",
                r"brute.*force",
                r"ddos",
                r"unauthorized.*access"
            ],
            ThreatLevel.MEDIUM: [
                r"failed.*login",
                r"suspicious.*activity",
                r"unusual.*traffic",
                r"port.*scan",
                r"probe",
                r"attempted.*access"
            ],
            ThreatLevel.LOW: [
                r"warning",
                r"notice",
                r"info",
                r"debug"
            ]
        }

        # Initialize threat detection results
        has_threat = False
        detected_threat_level = ThreatLevel.INFO
        event_type = EventType.UNKNOWN
        indicators = []

        # Check for event type
        if "event_type" in data:
            try:
                event_type = EventType(data["event_type"].lower())
            except ValueError:
                event_type = EventType.UNKNOWN

        # Analyze data based on event type
        if event_type == EventType.LOGIN:
            has_threat, detected_threat_level, indicators = self._analyze_login_events(data)
        elif event_type == EventType.NETWORK:
            has_threat, detected_threat_level, indicators = self._analyze_network_events(data)
        elif event_type == EventType.SYSTEM:
            has_threat, detected_threat_level, indicators = self._analyze_system_events(data)
        elif event_type == EventType.SECURITY:
            has_threat, detected_threat_level, indicators = self._analyze_security_events(data)
        else:
            # Generic analysis for unknown event types
            has_threat, detected_threat_level, indicators = self._analyze_generic_events(data, threat_patterns)

        # Check for context-based threats
        context_threat = self._check_context_threats(data)
        if context_threat["has_threat"]:
            has_threat = True
            if context_threat["threat_level"].value > detected_threat_level.value:
                detected_threat_level = context_threat["threat_level"]
            indicators.extend(context_threat["indicators"])

        return {
            "has_threat": has_threat,
            "threat_level": detected_threat_level,
            "event_type": event_type.value,
            "indicators": indicators,
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_login_events(self, data: Dict[str, Any]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze login-related events for threats."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Check for failed login attempts
        if data.get("status") == "failed":
            has_threat = True
            threat_level = ThreatLevel.MEDIUM
            indicators.append("Failed login attempt detected")

            # Check for brute force patterns
            recent_failures = self._count_recent_failures(data.get("user", ""))
            if recent_failures > 5:
                threat_level = ThreatLevel.HIGH
                indicators.append(f"Multiple failed login attempts ({recent_failures})")

        # Check for unusual login times
        if "timestamp" in data:
            login_time = datetime.fromisoformat(data["timestamp"]).time()
            if not self._is_business_hours(login_time):
                indicators.append("Login attempt outside business hours")

        return has_threat, threat_level, indicators

    def _analyze_network_events(self, data: Dict[str, Any]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze network-related events for threats."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Check for unusual traffic patterns
        if "traffic_volume" in data:
            if data["traffic_volume"] > 1000:  # Example threshold
                has_threat = True
                threat_level = ThreatLevel.MEDIUM
                indicators.append("Unusual network traffic volume detected")

        # Check for suspicious ports
        if "port" in data:
            suspicious_ports = {21, 23, 3389, 445}  # Example suspicious ports
            if data["port"] in suspicious_ports:
                has_threat = True
                threat_level = ThreatLevel.HIGH
                indicators.append(f"Suspicious port access detected: {data['port']}")

        return has_threat, threat_level, indicators

    def _analyze_system_events(self, data: Dict[str, Any]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze system-related events for threats."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Check for system changes
        if "change_type" in data:
            if data["change_type"] in ["file_modification", "config_change"]:
                has_threat = True
                threat_level = ThreatLevel.MEDIUM
                indicators.append(f"Suspicious system change detected: {data['change_type']}")

        # Check for resource usage
        if "cpu_usage" in data and "memory_usage" in data:
            if data["cpu_usage"] > 90 or data["memory_usage"] > 90:
                has_threat = True
                threat_level = ThreatLevel.MEDIUM
                indicators.append("High resource usage detected")

        return has_threat, threat_level, indicators

    def _analyze_security_events(self, data: Dict[str, Any]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze security-related events for threats."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Check for security alerts
        if "alert_type" in data:
            alert_severity = {
                "critical": ThreatLevel.CRITICAL,
                "high": ThreatLevel.HIGH,
                "medium": ThreatLevel.MEDIUM,
                "low": ThreatLevel.LOW
            }.get(data["alert_type"].lower(), ThreatLevel.INFO)

            if alert_severity != ThreatLevel.INFO:
                has_threat = True
                threat_level = alert_severity
                indicators.append(f"Security alert received: {data['alert_type']}")

        return has_threat, threat_level, indicators

    def _analyze_generic_events(self, data: Dict[str, Any], threat_patterns: Dict[ThreatLevel, List[str]]) -> tuple[bool, ThreatLevel, List[str]]:
        """Analyze generic events for threats using pattern matching."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Convert data to string for pattern matching
        data_str = str(data).lower()

        # Check each threat level's patterns
        for level, patterns in threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, data_str):
                    has_threat = True
                    if level.value > threat_level.value:
                        threat_level = level
                    indicators.append(f"Pattern match: {pattern}")

        return has_threat, threat_level, indicators

    def _check_context_threats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for threats based on context and historical data."""
        has_threat = False
        threat_level = ThreatLevel.INFO
        indicators = []

        # Check for rapid succession of events
        if len(self.event_history) > 1:
            last_event = self.event_history[-1]
            current_time = datetime.now()
            last_time = datetime.fromisoformat(last_event["timestamp"])
            time_diff = (current_time - last_time).total_seconds()

            if time_diff < 1:  # Events less than 1 second apart
                has_threat = True
                threat_level = ThreatLevel.MEDIUM
                indicators.append("Rapid succession of events detected")

        # Check for unusual patterns in event history
        if len(self.event_history) > 10:
            recent_events = self.event_history[-10:]
            if all(event["data"].get("status") == "failed" for event in recent_events):
                has_threat = True
                threat_level = ThreatLevel.HIGH
                indicators.append("Pattern of consistent failures detected")

        return {
            "has_threat": has_threat,
            "threat_level": threat_level,
            "indicators": indicators
        }

    def _count_recent_failures(self, username: str) -> int:
        """Count recent failed login attempts for a user."""
        if not username:
            return 0
        
        recent_events = self.event_history[-20:]  # Look at last 20 events
        return sum(1 for event in recent_events 
                  if event["data"].get("user") == username 
                  and event["data"].get("status") == "failed")

    def _is_business_hours(self, current_time: time) -> bool:
        """Check if the current time is within business hours."""
        return self.business_hours["start"] <= current_time <= self.business_hours["end"]

    def _generate_defensive_actions(self) -> Dict[str, Any]:
        """
        Generate appropriate defensive actions based on threat level.

        Returns:
            Dictionary containing recommended defensive actions
        """
        # Define actions for each threat level
        action_sets = {
            ThreatLevel.CRITICAL: {
                "primary_action": "isolate_system",
                "reason": "Critical threat detected",
                "additional_actions": [
                    "block_all_access",
                    "initiate_incident_response",
                    "notify_security_team",
                    "backup_critical_data",
                    "activate_emergency_protocols"
                ],
                "priority": 1
            },
            ThreatLevel.HIGH: {
                "primary_action": "quarantine_system",
                "reason": "High threat level detected",
                "additional_actions": [
                    "block_suspicious_ips",
                    "disable_affected_accounts",
                    "notify_security_team",
                    "increase_monitoring",
                    "review_security_logs"
                ],
                "priority": 2
            },
            ThreatLevel.MEDIUM: {
                "primary_action": "increase_monitoring",
                "reason": "Medium threat level detected",
                "additional_actions": [
                    "log_event",
                    "review_security_logs",
                    "update_firewall_rules",
                    "scan_system"
                ],
                "priority": 3
            },
            ThreatLevel.LOW: {
                "primary_action": "continue_monitoring",
                "reason": "Low threat level",
                "additional_actions": [
                    "log_event",
                    "update_threat_intelligence"
                ],
                "priority": 4
            },
            ThreatLevel.INFO: {
                "primary_action": "monitor",
                "reason": "Information only",
                "additional_actions": ["log_event"],
                "priority": 5
            }
        }

        # Get the appropriate action set based on threat level
        action_set = action_sets[self.threat_level]

        # Add context-specific actions based on event type
        if self.last_analysis and "event_type" in self.last_analysis:
            event_type = self.last_analysis["event_type"]
            if event_type == EventType.LOGIN.value:
                action_set["additional_actions"].extend([
                    "review_auth_logs",
                    "check_user_permissions"
                ])
            elif event_type == EventType.NETWORK.value:
                action_set["additional_actions"].extend([
                    "analyze_network_traffic",
                    "update_network_security"
                ])
            elif event_type == EventType.SYSTEM.value:
                action_set["additional_actions"].extend([
                    "check_system_integrity",
                    "review_system_changes"
                ])
            elif event_type == EventType.SECURITY.value:
                action_set["additional_actions"].extend([
                    "update_security_policies",
                    "review_security_configurations"
                ])

        return action_set 