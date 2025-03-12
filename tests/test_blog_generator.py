import pytest

from cybersec_agents import CyberSecurityBlogGenerator


class TestCyberSecurityBlogGenerator:
    """Test suite for CyberSecurityBlogGenerator."""

    @pytest.fixture
    def blog_generator(self, config):
        """Creates a CyberSecurityBlogGenerator instance."""
        return CyberSecurityBlogGenerator(config)

    @pytest.fixture
    def sample_topic(self):
        """Sample blog topic data."""
        return {
            "title": "Zero Trust Architecture Implementation",
            "keywords": ["zero trust", "security", "network segmentation"],
            "target_audience": "security professionals",
            "technical_level": "advanced",
        }

    def test_initialization(self, blog_generator):
        """Test proper initialization of the generator."""
        assert blog_generator.config is not None
        assert blog_generator.model is not None
        assert hasattr(blog_generator, "generate_blog_post")
        assert hasattr(blog_generator, "optimize_content")

    def test_generate_blog_post(self, blog_generator, sample_topic):
        """Test blog post generation."""
        result = blog_generator.generate_blog_post(topic=sample_topic, word_count=1500)

        assert isinstance(result, dict)
        assert "title" in result
        assert "content" in result
        assert "meta_description" in result
        assert "keywords" in result
        assert len(result["content"].split()) >= 1000

    def test_optimize_content(self, blog_generator):
        """Test content optimization for SEO."""
        content = {
            "title": "Basic Security Practices",
            "content": "This is a sample blog post about security...",
            "keywords": ["security", "best practices"],
        }

        result = blog_generator.optimize_content(content)

        assert isinstance(result, dict)
        assert "optimized_title" in result
        assert "optimized_content" in result
        assert "meta_tags" in result
        assert "keyword_density" in result

    def test_generate_technical_diagrams(self, blog_generator):
        """Test technical diagram generation."""
        diagram_spec = {
            "type": "network_architecture",
            "components": ["firewall", "dmz", "internal_network"],
            "style": "technical",
        }

        result = blog_generator.generate_technical_diagram(diagram_spec)

        assert isinstance(result, dict)
        assert "diagram_code" in result
        assert "format" in result
        assert "elements" in result

    def test_validate_technical_accuracy(self, blog_generator):
        """Test technical content validation."""
        content = {
            "title": "Understanding Buffer Overflows",
            "content": "Technical content about buffer overflows...",
            "technical_claims": [
                "Stack-based buffer overflows can lead to code execution",
                "DEP prevents direct code execution in the stack",
            ],
        }

        result = blog_generator.validate_technical_accuracy(content)

        assert isinstance(result, dict)
        assert "accuracy_score" in result
        assert "verified_claims" in result
        assert "corrections_needed" in result

    def test_generate_code_samples(self, blog_generator):
        """Test code sample generation."""
        context = {
            "language": "python",
            "topic": "input validation",
            "security_focus": "preventing SQL injection",
        }

        result = blog_generator.generate_code_samples(context)

        assert isinstance(result, dict)
        assert "code_snippets" in result
        assert "explanations" in result
        assert "security_notes" in result

    def test_monetization_integration(self, blog_generator):
        """Test monetization feature integration."""
        content = {
            "title": "Advanced Penetration Testing Tools",
            "content": "Detailed content about pen testing...",
            "monetization_strategy": "premium_content",
        }

        result = blog_generator.apply_monetization(content)

        assert isinstance(result, dict)
        assert "premium_sections" in result
        assert "preview_content" in result
        assert "subscription_cta" in result

    def test_generate_series_outline(self, blog_generator):
        """Test blog series outline generation."""
        series_topic = {
            "main_topic": "Web Application Security",
            "num_posts": 5,
            "technical_level": "intermediate",
        }

        result = blog_generator.generate_series_outline(series_topic)

        assert isinstance(result, dict)
        assert "series_title" in result
        assert "posts" in result
        assert len(result["posts"]) == 5
        assert "learning_path" in result

    def test_invalid_topic(self, blog_generator):
        """Test error handling for invalid topic."""
        with pytest.raises(ValueError) as exc_info:
            blog_generator.generate_blog_post(topic={}, word_count=1500)
        assert "Invalid topic specification" in str(exc_info.value)

    def test_content_compliance(self, blog_generator, sample_topic):
        """Test content compliance checking."""
        result = blog_generator.check_content_compliance(
            topic=sample_topic, compliance_standards=["gdpr", "ccpa"]
        )

        assert isinstance(result, dict)
        assert "compliant" in result
        assert "violations" in result
        assert "recommendations" in result

    def test_generate_social_media_content(self, blog_generator):
        """Test social media content generation."""
        blog_post = {
            "title": "Zero Trust Architecture",
            "content": "Detailed technical content...",
            "key_points": ["point1", "point2"],
        }

        result = blog_generator.generate_social_media_content(blog_post)

        assert isinstance(result, dict)
        assert "twitter_posts" in result
        assert "linkedin_post" in result
        assert "hashtags" in result
