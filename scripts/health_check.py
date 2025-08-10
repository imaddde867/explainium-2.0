#!/usr/bin/env python3
"""
EXPLAINIUM Health Check Script

Comprehensive health monitoring for production deployment.
"""

import requests
import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import subprocess

class HealthChecker:
    """Professional health monitoring for EXPLAINIUM"""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.frontend_base = "http://localhost:8501"
        self.tika_base = "http://localhost:9998"
        self.redis_host = "localhost"
        self.redis_port = 6379
        
    def check_api_health(self) -> Dict[str, Any]:
        """Check FastAPI backend health"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "data": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time": response.elapsed.total_seconds()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": None
            }
    
    def check_frontend_health(self) -> Dict[str, Any]:
        """Check Streamlit frontend health"""
        try:
            response = requests.get(self.frontend_base, timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time": response.elapsed.total_seconds()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": None
            }
    
    def check_tika_health(self) -> Dict[str, Any]:
        """Check Apache Tika service health"""
        try:
            response = requests.get(f"{self.tika_base}/tika", timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time": response.elapsed.total_seconds()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": None
            }
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis service health"""
        try:
            import redis
            r = redis.Redis(host=self.redis_host, port=self.redis_port, socket_timeout=5)
            r.ping()
            return {
                "status": "healthy",
                "info": r.info()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_celery_workers(self) -> Dict[str, Any]:
        """Check Celery worker status"""
        try:
            response = requests.get(f"{self.api_base}/workers/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "workers": data
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    return {
                        "status": "healthy",
                        "total": fields[1],
                        "used": fields[2],
                        "available": fields[3],
                        "usage_percent": fields[4]
                    }
            return {"status": "error", "error": "Could not parse df output"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    return {
                        "status": "healthy",
                        "total": fields[1],
                        "used": fields[2],
                        "free": fields[3],
                        "available": fields[6] if len(fields) > 6 else fields[3]
                    }
            return {"status": "error", "error": "Could not parse free output"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        print("🔍 Running EXPLAINIUM Health Check...")
        print("=" * 50)
        
        checks = {
            "api": self.check_api_health(),
            "frontend": self.check_frontend_health(),
            "tika": self.check_tika_health(),
            "redis": self.check_redis_health(),
            "celery": self.check_celery_workers(),
            "disk": self.check_disk_space(),
            "memory": self.check_memory_usage()
        }
        
        # Print results
        overall_status = "healthy"
        for service, result in checks.items():
            status = result.get("status", "unknown")
            if status == "healthy":
                print(f"✅ {service.upper()}: Healthy")
                if "response_time" in result:
                    print(f"   Response time: {result['response_time']:.3f}s")
            elif status == "unhealthy":
                print(f"⚠️  {service.upper()}: Unhealthy - {result.get('error', 'Unknown error')}")
                overall_status = "degraded"
            else:
                print(f"❌ {service.upper()}: Error - {result.get('error', 'Unknown error')}")
                overall_status = "unhealthy"
        
        print("=" * 50)
        
        if overall_status == "healthy":
            print("🎉 All systems are healthy!")
        elif overall_status == "degraded":
            print("⚠️  Some services are degraded")
        else:
            print("❌ Critical issues detected")
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": checks
        }


def main():
    """Main health check execution"""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # JSON output for monitoring systems
        result = checker.run_comprehensive_check()
        print(json.dumps(result, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        # Continuous monitoring mode
        print("🔄 Starting continuous health monitoring...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                result = checker.run_comprehensive_check()
                if result["overall_status"] != "healthy":
                    print(f"⚠️  Issues detected at {result['timestamp']}")
                
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\n👋 Health monitoring stopped")
    else:
        # Single check
        result = checker.run_comprehensive_check()
        sys.exit(0 if result["overall_status"] == "healthy" else 1)


if __name__ == "__main__":
    main()