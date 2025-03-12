from unittest.mock import patch

import pytest

from cybersec_agents import CodebaseAnalyzerAgent


class TestCodebaseAnalyzerAgent:
    """Test suite for CodebaseAnalyzerAgent."""

    @pytest.fixture
    def sample_python_code(self):
        return """
def process_user_input(data):
    query = f"SELECT * FROM users WHERE id = {data['user_id']}"
    cursor.execute(query)
    return cursor.fetchall()

def validate_token(token):
    if token == "secret_token":
        return True
    return False
        """

    @pytest.fixture
    def sample_javascript_code(self):
        return """
function handleUserData(userData) {
    const token = userData.token;
    if (token === 'admin') {
        eval(userData.command);
    }
    localStorage.setItem('userToken', token);
}
        """

    @pytest.fixture
    def analyzer(self, config):
        """Creates a CodebaseAnalyzerAgent instance."""
        return CodebaseAnalyzerAgent(config)

    def test_initialization(self, analyzer):
        """Test proper initialization of the analyzer."""
        assert analyzer.config is not None
        assert analyzer.model is not None
        assert hasattr(analyzer, "analyze_code")
        assert hasattr(analyzer, "scan_codebase")

    def test_analyze_python_code(self, analyzer, sample_python_code):
        """Test Python code analysis for security vulnerabilities."""
        result = analyzer.analyze_code(code=sample_python_code, language="python")

        assert isinstance(result, dict)
        assert "vulnerabilities" in result
        assert "recommendations" in result

        vulns = result["vulnerabilities"]
        assert any(v["type"] == "SQL_INJECTION" for v in vulns)
        assert any(v["type"] == "HARDCODED_SECRET" for v in vulns)

    def test_analyze_javascript_code(self, analyzer, sample_javascript_code):
        """Test JavaScript code analysis for security vulnerabilities."""
        result = analyzer.analyze_code(
            code=sample_javascript_code, language="javascript"
        )

        assert isinstance(result, dict)
        assert "vulnerabilities" in result
        assert "recommendations" in result

        vulns = result["vulnerabilities"]
        assert any(v["type"] == "EVAL_USAGE" for v in vulns)
        assert any(v["type"] == "INSECURE_DATA_STORAGE" for v in vulns)

    @patch("cybersec_agents.CodebaseAnalyzerAgent._scan_directory")
    def test_scan_codebase(self, mock_scan, analyzer, tmp_path):
        """Test scanning entire codebase."""
        mock_scan.return_value = {
            "files_analyzed": 10,
            "vulnerabilities_found": [
                {
                    "file": "src/auth.py",
                    "line": 15,
                    "type": "WEAK_CRYPTO",
                    "severity": "HIGH",
                }
            ],
            "security_score": 75,
        }

        result = analyzer.scan_codebase(
            path=tmp_path, exclude_patterns=["*.test.js", "*.spec.py"]
        )

        assert isinstance(result, dict)
        assert "files_analyzed" in result
        assert "vulnerabilities_found" in result
        assert "security_score" in result
        assert 0 <= result["security_score"] <= 100

    def test_analyze_dependencies(self, analyzer):
        """Test analysis of project dependencies."""
        requirements = """
        django==2.2.0
        requests==2.25.0
        cryptography==3.2
        """

        result = analyzer.analyze_dependencies(requirements)

        assert isinstance(result, dict)
        assert "vulnerable_packages" in result
        assert "outdated_packages" in result
        assert "recommendations" in result

    def test_generate_security_report(self, analyzer):
        """Test security report generation."""
        scan_results = {
            "vulnerabilities": [
                {"file": "src/auth.py", "type": "WEAK_CRYPTO", "severity": "HIGH"}
            ],
            "security_score": 75,
        }

        report = analyzer.generate_security_report(scan_results)

        assert isinstance(report, dict)
        assert "summary" in report
        assert "detailed_findings" in report
        assert "remediation_steps" in report
        assert "risk_assessment" in report

    def test_analyze_commit_history(self, analyzer):
        """Test analysis of git commit history for security issues."""
        commits = [
            {
                "hash": "abc123",
                "message": "Added API key for testing",
                "date": "2024-02-01",
                "author": "developer@example.com",
            }
        ]

        result = analyzer.analyze_commit_history(commits)

        assert isinstance(result, dict)
        assert "sensitive_commits" in result
        assert "risk_patterns" in result
        assert "recommendations" in result

    def test_analyze_configuration_files(self, analyzer):
        """Test analysis of configuration files."""
        config_files = {
            "config/database.yml": """
                production:
                    password: secret123
                    host: db.example.com
            """,
            "docker-compose.yml": """
                services:
                    web:
                        environment:
                            - API_KEY=abc123
            """,
        }

        result = analyzer.analyze_configuration_files(config_files)

        assert isinstance(result, dict)
        assert "exposed_secrets" in result
        assert "security_misconfigurations" in result
        assert "recommendations" in result

    def test_invalid_code_input(self, analyzer):
        """Test error handling for invalid code input."""
        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_code(code="", language="python")
        assert "Invalid code input" in str(exc_info.value)

    def test_unsupported_language(self, analyzer, sample_python_code):
        """Test error handling for unsupported programming language."""
        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_code(code=sample_python_code, language="brainfuck")
        assert "Unsupported programming language" in str(exc_info.value)

    @patch("cybersec_agents.CodebaseAnalyzerAgent._analyze_file")
    def test_incremental_analysis(self, mock_analyze, analyzer):
        """Test incremental analysis of changed files."""
        changed_files = ["src/auth.py", "src/api/endpoints.py"]

        mock_analyze.return_value = {"vulnerabilities": [], "security_score": 90}

        result = analyzer.analyze_changed_files(changed_files)

        assert isinstance(result, dict)
        assert "files_analyzed" in result
        assert "new_vulnerabilities" in result
        assert "resolved_vulnerabilities" in result
        assert "security_score_delta" in result

    def test_custom_rule_addition(self, analyzer):
        """Test adding custom security rules."""
        custom_rule = {
            "id": "CUSTOM_RULE_001",
            "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
            "severity": "HIGH",
            "description": "Hardcoded password detected",
        }

        analyzer.add_custom_rule(custom_rule)

        test_code = "password = 'secret123'"
        result = analyzer.analyze_code(code=test_code, language="python")

        assert any(v["rule_id"] == "CUSTOM_RULE_001" for v in result["vulnerabilities"])
