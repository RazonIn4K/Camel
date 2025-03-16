import asyncio
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, patch

import pytest

from cybersec_agents import NetworkAnomalyDetector
from cybersec_agents.config import Config


class TestPerformance:
    @pytest.fixture
    def config(self):
        return Config()

    @pytest.fixture
    def detector(self, config):
        return NetworkAnomalyDetector(config)

    @pytest.fixture
    def large_nmap_xml(self):
        """Generate a large Nmap XML file with multiple hosts and ports."""
        root = ET.Element("nmaprun")
        for i in range(100):  # 100 hosts
            host = ET.SubElement(root, "host")
            addr = ET.SubElement(host, "address")
            addr.set("addr", f"192.168.1.{i}")
            addr.set("addrtype", "ipv4")

            ports = ET.SubElement(host, "ports")
            for port in range(20):  # 20 ports per host
                port_elem = ET.SubElement(ports, "port")
                port_elem.set("protocol", "tcp")
                port_elem.set("portid", str(port + 1))

                state = ET.SubElement(port_elem, "state")
                state.set("state", "open")

                service = ET.SubElement(port_elem, "service")
                service.set("name", f"service_{port}")

        tree = ET.ElementTree(root)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False) as f:
            tree.write(f)
            return Path(f.name)

    async def measure_execution_time(self, coro):
        """Measure execution time of a coroutine."""
        start_time = time.perf_counter()
        await coro
        end_time = time.perf_counter()
        return end_time - start_time

    @pytest.mark.asyncio
    async def test_analysis_performance(self, detector, large_nmap_xml):
        """Test performance of the analysis pipeline."""
        mock_response: dict[str, Any] = {
            "threats": ["Sample threat"],
            "vulnerabilities": ["Sample vulnerability"],
            "recommendations": ["Sample recommendation"],
            "risk_level": "MEDIUM",
        }

        with patch.object(detector.agent, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            # Warm-up run
            await detector.analyze_nmap_output(large_nmap_xml)

            # Measure multiple runs
            times: list[Any] = []
            for _ in range(5):
                execution_time = await self.measure_execution_time(
                    detector.analyze_nmap_output(large_nmap_xml)
                )
                times.append(execution_time)

            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile

            # Performance assertions
            assert (
                avg_time < 2.0
            ), f"Average execution time ({avg_time:.2f}s) exceeded 2.0s threshold"
            assert (
                p95_time < 3.0
            ), f"95th percentile time ({p95_time:.2f}s) exceeded 3.0s threshold"

    @pytest.mark.asyncio
    async def test_concurrent_performance(self, detector, large_nmap_xml):
        """Test performance under concurrent load."""
        mock_response: dict[str, Any] = {
            "threats": ["Sample threat"],
            "vulnerabilities": ["Sample vulnerability"],
            "recommendations": ["Sample recommendation"],
            "risk_level": "MEDIUM",
        }

        with patch.object(detector.agent, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            # Create multiple concurrent requests
            concurrent_requests: int = 5
            start_time = time.perf_counter()

            tasks: list[Any] = [
                detector.analyze_nmap_output(large_nmap_xml)
                for _ in range(concurrent_requests)
            ]

            await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            # Calculate throughput
            throughput = concurrent_requests / total_time
            assert (
                throughput >= 1.0
            ), f"Throughput ({throughput:.2f} req/s) below 1.0 req/s threshold"

    def test_config_load_performance(self):
        """Test configuration loading performance."""
        times: list[Any] = []
        for _ in range(100):
            start_time = time.perf_counter()
            Config()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = statistics.mean(times)
        assert (
            avg_time < 0.01
        ), f"Average config loading time ({avg_time:.4f}s) exceeded 10ms threshold"

    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self, detector):
        """Test rate limiter performance under load."""
        start_time = time.perf_counter()
        request_count: int = 100
        success_count: int = 0

        for _ in range(request_count):
            try:
                await detector.rate_limiter.acquire()
                success_count += 1
            except:
                pass

        total_time = time.perf_counter() - start_time
        rate = success_count / total_time

        assert (
            rate <= detector.config.agent.rate_limit * 1.1
        ), f"Rate limiter allowing too many requests ({rate:.2f} req/s)"

    @pytest.mark.benchmark
    def test_file_validation_performance(self, detector, benchmark):
        """Benchmark file validation performance."""

        def create_test_file(size_mb):
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
                f.write(b"0" * (size_mb * 1024 * 1024))
                return Path(f.name)

        test_file = create_test_file(1)  # 1MB file

        def validate_file():
            detector.file_validator.validate(test_file)

        # Run benchmark
        result: Any = benchmark(validate_file)

        # Cleanup
        test_file.unlink()

        assert (
            result.stats.mean < 0.01
        ), f"File validation too slow: {result.stats.mean:.4f}s average"

    @pytest.mark.asyncio
    async def test_memory_usage(self, detector, large_nmap_xml):
        """Test memory usage during analysis."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        mock_response: dict[str, Any] = {
            "threats": ["Sample threat"],
            "vulnerabilities": ["Sample vulnerability"],
            "recommendations": ["Sample recommendation"],
            "risk_level": "MEDIUM",
        }

        with patch.object(detector.agent, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response

            # Run multiple analyses
            for _ in range(10):
                await detector.analyze_nmap_output(large_nmap_xml)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert (
            memory_increase < 50
        ), f"Memory usage increased by {memory_increase:.1f}MB, exceeding 50MB threshold"
